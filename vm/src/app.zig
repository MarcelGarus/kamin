const Value = @import("value.zig");
const object_loader = @import("object_loader.zig");
const Heap = @import("heap.zig");
const Vm = @import("vm.zig");
const Graphics = @import("graphics.zig");
const Obj = Heap.Obj;

const std = @import("std");
const Ally = std.mem.Allocator;
const Io = std.Io;

// Similar to Smalltalk, I want all my data to be part of a value / an "image"
// that can evolve over time. Unlike Smalltalk, I want to support multiple
// clients working on the same value. For example, I might have a version of
// Orchard on my laptop and my phone and data (notes, calendar events, ...)
// should sync between them.
//
// To do that, I separate two pieces of state:
// - Global state that is syncronized between devices.
// - Local state that is per-client.
//
// Both the global store and the local state are just values, so it's easy to
// move a piece from the local to the global state.
//
// If Orchard instances would just proclaim "this is the new global state", then
// it would be difficult to merge updates (for example, if there's some network
// latency so that concurrent updates happen). To fix that, Orchard instances
// only emit actions ("add avocado to the shopping list"), which can be applied
// to the state. Then, multiple clients can work on the same global state.
//
// +-------+
// | store | <-----> Orchard instance on laptop
// |       | <-----> Orchard instance on phone
// |       | <-----> headless daemon that fetches e.g. RSS feeds
// +-------+
//
// There are multiple ways of doing synchronization (including peer-to-peer),
// but here's a simpler server-client scenario:
//
// | Our client fetches some server state S1 and starts with it.
// S1
// | Perform actions A and B: Send them to the server. Also eagerly apply them
// | locally so that everything feels instantaneous.
// S1 + A + B
// | The server sends us A, so we know it has been "committed" to the server
// | state. New server state S2 = S1 + A.
// S2 + B
// | The server sends us X, an action by another device. We use S3 = S2 + X as
// | the new baseline of our history, applying all uncommitted changes to that.
// S3 + B
// | The server sends us B. S4 = S3 + B.
// S4
//
// In the end, the server serializes actions (A, X, B) and this is kept
// consistent between all clients.

// TODO: For now, I don't do any synchronization yet. I just read the global
// state from a file and when an action is applied, I calculate the new state
// and write it into the file.
const FileStore = struct {
    path: []const u8,
    apply: Value, // (state, action) -> state
    state: Value,

    fn init(ally: Ally, io: Io, heap: *Heap, path: []const u8, apply: Value) !@This() {
        return .{
            .path = path,
            .apply = apply,
            .state = Value.from(try object_loader.load(
                ally,
                heap,
                try Io.Dir.cwd().readFileAlloc(io, path, ally, .unlimited),
            )),
        };
    }
    fn add_action(self: *@This(), ally: Ally, io: Io, vm: anytype, action: Value) !void {
        self.state = try self.apply.call(vm, &.{ self.state, action });
        // std.debug.print("Action: {f}\n", .{action});
        // std.debug.print("global state: {f}\n", .{self.state});

        const out = try std.Io.Dir.cwd().createFile(io, self.path, .{});
        var out_buf: [1000]u8 = undefined;
        var out_w = out.writer(io, &out_buf);
        try Heap.file_out(self.state.obj, ally, &out_w.interface);
    }
};

// Given some changing data source, app instances work like this:
//
// +--------------+             +--------------+             +----------------------+
// | global state | --create--> | local state  | --render--> | drawing instructions |
// +--------------+             +--------------+             +----------------------+
//                                |          ^
// +------------------------+     |          |
// | events                 | ----+--handle--+
// | - key presses          |
// | - global state updates |
// | - ...                  |
// +------------------------+

value: Value,

const App = @This();

const Handle = usize;
// A Handle is an integer that references a resource living outside the VM.

const AppEvent = union(enum) {
    window_resized: Graphics.Size,
    key_pressed: Graphics.Event.KeyPressed,
    char_entered: Graphics.Event.CharEntered,
    mouse_clicked: Graphics.Event.MouseClicked,
    task_finished: *Task,
};

const EventQueue = struct {
    ally: Ally,
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    events: std.Deque(AppEvent) = .empty,

    fn init(ally: Ally) EventQueue {
        return .{ .ally = ally };
    }
    fn deinit(self: *EventQueue) void {
        self.events.deinit(self.ally);
    }

    fn acquire(self: *EventQueue) void {
        while (self.locked.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    fn release(self: *EventQueue) void {
        self.locked.store(false, .release);
    }

    fn push(self: *EventQueue, event: AppEvent) void {
        self.acquire();
        self.events.pushBack(self.ally, event) catch @panic("out of memory pushing app event");
        self.release();
        Graphics.wake();
    }

    fn next(self: *EventQueue) ?AppEvent {
        self.acquire();
        defer self.release();
        return self.events.popFront();
    }
};

fn event_to_value(heap: *Heap, event: AppEvent) !Value {
    return switch (event) {
        .char_entered => |char| try Value.new_enum(heap, "char-entered", ch: {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(@intCast(char.codepoint), &buf) catch
                break :ch try Value.new_string(heap, "?");
            break :ch try Value.new_string(heap, buf[0..len]);
        }),
        .key_pressed => |key| try Value.new_enum(heap, "key-pressed", try Value.new_struct(heap, .{
            .keycode = try Value.new_int(heap, @intCast(key.keycode)),
            .control = try Value.new_bool(heap, key.control),
            .shift = try Value.new_bool(heap, key.shift),
            .alt = try Value.new_bool(heap, key.alt),
        })),
        .mouse_clicked => |position| try Value.new_enum(heap, "mouse-clicked", try Value.new_struct(heap, .{
            .x = try Value.new_float(heap, position.x),
            .y = try Value.new_float(heap, position.y),
        })),
        .window_resized => |size| try Value.new_enum(heap, "window-resized", try Value.new_struct(heap, .{
            .width = try Value.new_float(heap, size.width),
            .height = try Value.new_float(heap, size.height),
        })),
        .task_finished => unreachable,
    };
}

pub fn run(ally: Ally, io: Io, heap: *Heap, vm: anytype, app_: Value, data_file: []const u8) !void {
    _ = data_file;
    // var global = try FileStore.init(ally, io, heap, data_file, app_.field("apply"));
    // std.debug.print("global state: {f}\n", .{global.state});

    var app = App{ .value = app_ };

    var gfx = try Graphics.init(ally);
    defer gfx.deinit();

    // The app event queue drives everything. Subsystems push onto it: graphics
    // events are harvested here each frame, and background tasks push themselves
    // when they finish (see run_background_task).
    var queue = EventQueue.init(ally);
    defer queue.deinit();
    var next_handle: Handle = 1;

    // Cached drawing instructions for a particular size.
    var previous_size = gfx.get_size();
    var drawing_instructions: []const Graphics.DrawingInstruction = &.{};
    var drawing_instructions_ally: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(ally);
    queue.push(.{ .window_resized = previous_size }); // always render on the first frame

    while (!gfx.should_close()) {
        // Add events to our queue:

        // Add graphics events.
        for (gfx.event_queue.items) |event| {
            queue.push(switch (event) {
                .char_entered => |char| .{ .char_entered = char },
                .key_pressed => |key| .{ .key_pressed = key },
                .mouse_clicked => |position| .{ .mouse_clicked = position },
            });
        }
        gfx.event_queue.clearRetainingCapacity();
        // Add resize events.
        const size = gfx.get_size();
        if (size.width != previous_size.width or size.height != previous_size.height)
            queue.push(.{ .window_resized = size });
        previous_size = size;
        // Note: Task events are automatically added by the tasks.

        std.debug.print(".", .{});

        // If there were no events since the last frame, the VM will produce the
        // exact same drawing instructions. Just use the cached result.
        if (queue.events.len == 0) {
            // try gfx.render(size, drawing_instructions);
            gfx.poll_events();
            continue;
        }

        std.debug.print("loop\n", .{});
        const loop_start = Io.Timestamp.now(io, .real);

        // Make the VM process the events.
        std.debug.print("  {} events\n", .{queue.events.len});
        while (queue.next()) |event| {
            const event_start = Io.Timestamp.now(io, .real);
            const event_value = switch (event) {
                .task_finished => |task| ev: {
                    task.thread.join(); // task has already pushed; join reaps it.
                    const result = try import_task_result(ally, heap, task);
                    const v = try Value.new_enum(heap, "task-finished", try Value.new_struct(heap, .{
                        .handle = try Value.new_int(heap, @intCast(task.handle)),
                        .result = result,
                    }));
                    task.arena.deinit();
                    ally.destroy(task);
                    break :ev v;
                },
                else => try event_to_value(heap, event),
            };
            std.debug.print("    {f}\n", .{event_value});

            // Feed the event value to the app and resolve the resulting chain
            // of effects until the app yields a new version of itself.
            //
            // `handle` returns an effect enum; the host acts on it and continues:
            // (| app newapp) -> returns the new app.
            // (| spawn-task (& task ... then (\ (handle) ...))) -> starts a background task; continues with then.
            // (| cancel-task (& pid .. then (\ () ..))) -> cancels task; continues with then.
            var effect = try app.value.field("handle").call(vm, &.{event_value});
            while (true) {
                const effect_start = Io.Timestamp.now(io, .real);
                const v = effect.variant();
                std.debug.print("      {f}", .{effect});
                if (std.mem.eql(u8, v, "app")) {
                    app = .{ .value = effect.payload() };
                    std.debug.print("\n", .{});
                    break;
                } else if (std.mem.eql(u8, v, "spawn-task")) {
                    const payload = effect.payload();
                    const handle = next_handle;
                    next_handle += 1;
                    try run_background_task(&queue, heap, ally, handle, payload.field("task").obj);
                    effect = try payload.field("then").call(vm, &.{try Value.new_int(heap, @intCast(handle))});
                    std.debug.print(" {} ms \n", .{effect_start.untilNow(io, .real).toMilliseconds()});
                } else if (std.mem.eql(u8, v, "cancel-task")) {
                    const payload = effect.payload();
                    effect = try payload.field("then").call(vm, &.{});
                    std.debug.print(" {} ms \n", .{effect_start.untilNow(io, .real).toMilliseconds()});
                } else {
                    std.debug.print("Unknown effect from handle: {s}\n", .{v});
                    @panic("unknown effect variant");
                }
            }
            std.debug.print("      {} ms\n", .{event_start.untilNow(io, .real).toMilliseconds()});
        }
        std.debug.print("    {} ms\n", .{loop_start.untilNow(io, .real).toMilliseconds()});

        std.debug.print("  rendering:", .{});
        const render_start = Io.Timestamp.now(io, .real);
        drawing_instructions_ally.deinit();
        drawing_instructions_ally = std.heap.ArenaAllocator.init(ally);
        drawing_instructions = render_to_instructions: {
            const frame_checkpoint = heap.checkpoint();
            defer heap.restore(frame_checkpoint);

            const instructions = try app.value.field("render").call(vm, &[_]Value{});
            const render_vm_end = Io.Timestamp.now(io, .real);
            std.debug.print(" vm {} ms", .{(render_start.durationTo(render_vm_end)).toMilliseconds()});

            const parsed = try Graphics.parse_drawing_instructions(drawing_instructions_ally.allocator(), instructions);
            const render_parse_end = Io.Timestamp.now(io, .real);
            std.debug.print(", parse {} ms", .{(render_vm_end.durationTo(render_parse_end)).toMilliseconds()});
            break :render_to_instructions parsed;
        };
        const render_end = Io.Timestamp.now(io, .real);
        try gfx.render(size, drawing_instructions);
        std.debug.print(", raylib {} ms", .{(render_end.untilNow(io, .real)).toMilliseconds()});
        std.debug.print(", total {} ms\n", .{render_start.untilNow(io, .real).toMilliseconds()});

        std.debug.print("  heap: ", .{});
        heap.dump_size();
        std.debug.print("\n", .{});

        gfx.poll_events();
    }
}

// Imports a finished task's result out of its private worker heap into the main
// heap, as either the raw value or `(| crashed payload)`.
// TODO: fix this
fn import_task_result(ally: Ally, heap: *Heap, task: *Task) !Value {
    if (task.crashed) {
        const payload: Obj = if (task.result) |obj|
            try task.heap.copy_to_other_heap(ally, heap, obj)
        else
            (try Value.new_string(heap, "task failed")).obj;
        return try Value.new_enum(heap, "crashed", Value{ .obj = payload });
    }
    return Value{ .obj = try task.heap.copy_to_other_heap(ally, heap, task.result.?) };
}

// Spawns `thunk` (a 0-arg closure in the main heap) on a fresh background thread
// with its own fully-isolated heap and VM. The thread copies the thunk into its
// own heap up front, runs it, and pushes a `task_finished` event onto `queue`
// when done — so there is no manager and no shared state, just this function and
// the queue the worker reports back to.
fn run_background_task(queue: *EventQueue, main_heap: *Heap, main_ally: Ally, handle: Handle, thunk: Obj) !void {
    const task = try main_ally.create(Task);
    errdefer main_ally.destroy(task);
    task.arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    errdefer task.arena.deinit();
    const wally = task.arena.allocator();
    task.heap = try Heap.init(wally, WORKER_HEAP_WORDS);
    task.vm = try Vm.Default.init_sized(&task.heap, wally, WORKER_DATA_STACK_WORDS, WORKER_CALL_STACK_WORDS);

    // Copy the thunk's whole reachable subgraph out of the main heap into the
    // worker's own heap, so the worker is fully self-contained.
    task.thunk = try main_heap.copy_to_other_heap(main_ally, &task.heap, thunk);
    task.handle = handle;
    task.queue = queue;
    task.result = null;
    task.crashed = false;

    task.thread = try std.Thread.spawn(.{}, worker_main, .{task});
}

const WORKER_HEAP_WORDS = 8 * 1024 * 1024; // 64 MiB
const WORKER_DATA_STACK_WORDS = 2 * 1024 * 1024; // 16 MiB
const WORKER_CALL_STACK_WORDS = 2 * 1024 * 1024; // 16 MiB

// A background task: a thread with its own fully-isolated heap and VM. Owned by
// nobody centrally — it travels from the spawning main thread to its worker
// thread and back to the main thread via the `task_finished` event, which then
// imports the result and frees it. No manager, no shared list.
const Task = struct {
    handle: Handle,
    queue: *EventQueue,
    thread: std.Thread,
    arena: std.heap.ArenaAllocator,
    heap: Heap,
    vm: Vm.Default,
    thunk: Obj, // in the worker's own heap (copied in at spawn)
    result: ?Obj, // in the worker's heap
    crashed: bool,
};

const ThunkResult = union(enum) { ok: Obj, crash: Obj, failed };

// Runs a 0-arg closure on the given VM. The closure and its captured environment
// live in the VM's own heap; new objects land there too.
fn run_thunk(vm: *Vm.Default, thunk_obj: Obj) ThunkResult {
    const thunk = Value{ .obj = thunk_obj };
    const fun = thunk.vm_fun();
    const closure = thunk.captured().obj;
    var fuel: usize = Vm.max_fuel;
    const result = vm.call(fun, &.{closure}, &fuel) catch return .failed;
    return switch (result) {
        .returned => |obj| .{ .ok = obj },
        .crashed => |obj| .{ .crash = obj },
        .out_of_fuel, .out_of_memory => .failed,
    };
}
fn worker_main(task: *Task) void {
    switch (run_thunk(&task.vm, task.thunk)) {
        .ok => |obj| {
            task.result = gc_keep(task, obj);
            task.crashed = false;
        },
        .crash => |obj| {
            task.result = gc_keep(task, obj);
            task.crashed = true;
        },
        .failed => {
            task.result = null;
            task.crashed = true;
        },
    }
    task.queue.push(.{ .task_finished = task });
}
fn gc_keep(task: *Task, result: Obj) Obj {
    const gcd = task.heap.garbage_collect(task.arena.allocator(), task.heap.start(), result) catch return result;
    return gcd.keep;
}
