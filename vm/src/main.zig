const std = @import("std");
const object_loader = @import("object_loader.zig");
const Heap = @import("heap.zig");
const Vm = @import("vm.zig");
const Graphics = @import("graphics.zig");
const Value = @import("value.zig");
const Obj = Heap.Obj;
const Ally = std.mem.Allocator;
const Io = std.Io;
const App = @import("app.zig");

const BootstrapStep = struct {
    starttime: Io.Timestamp,
    heap: *Heap,

    fn start(io: Io, comptime name: []const u8, heap: *Heap) !BootstrapStep {
        std.debug.print(name, .{});
        for (name.len..30) |_| std.debug.print(" ", .{});
        const now = Io.Timestamp.now(io, .real);
        return .{ .starttime = now, .heap = heap };
    }
    fn end(self: *BootstrapStep, io: Io) void {
        const now = Io.Timestamp.now(io, .real);
        std.debug.print("{} ms, ", .{now.toMilliseconds() - self.starttime.toMilliseconds()});
        self.heap.dump_stats();
    }
};

fn is_same(a: Obj, b: Obj) bool {
    if (a.address == b.address) return true;
    if (a.is_inner() != b.is_inner()) return false;
    if (a.size() != b.size()) return false;
    if (a.is_inner()) {
        for (a.children(), b.children()) |ac, bc| if (!is_same(ac, bc)) return false;
    } else {
        for (a.words(), b.words()) |aw, bw| if (aw != bw) return false;
    }
    return true;
}
pub fn main(init: std.process.Init) !void {
    // std.debug.print("Orchard!\n", .{});

    const ally = init.gpa;
    const io = init.io;

    // const stdout_buf = try ally.alloc(u8, 1000);
    // const stdin_buf = try ally.alloc(u8, 1000);
    // var stdout_w = Io.File.stdout().writer(io, stdout_buf);
    // var stdin_r = Io.File.stdin().reader(io, stdin_buf);
    // const stdout = &stdout_w.interface;
    // const stdin = &stdin_r.interface;

    var heap = try Heap.init(ally, 600_000_000);
    const start_of_heap = heap.checkpoint();
    var vm = try Vm.Default.init(&heap, ally);

    const objects_code = try Io.Dir.cwd().readFileAlloc(io, "src/bootstrap.objects", ally, .unlimited);
    const olive_code = try Io.Dir.cwd().readFileAlloc(io, "src/bootstrap.olive", ally, .unlimited);
    const pear_code = try Io.Dir.cwd().readFileAlloc(io, "src/bootstrap.pear", ally, .unlimited);

    const compile_olive = step: {
        var step = try BootstrapStep.start(io, "Loading the Olive compiler.", &heap);
        defer step.end(io);
        break :step Value.from(try object_loader.load(ally, &heap, objects_code));
    };
    std.debug.print("objects:\n{f}\n", .{compile_olive.obj});
    const olive = step: {
        var step = try BootstrapStep.start(io, "Compiling Olive.", &heap);
        defer step.end(io);
        const result = try compile_olive.call(&vm, &.{
            try Value.new_string(&heap, olive_code),
        });
        break :step Value.from(try vm.garbage_collect(start_of_heap, result.obj));
    };
    const olive_self_hosted = step: {
        var step = try BootstrapStep.start(io, "Self-compiling Olive.", &heap);
        defer step.end(io);
        break :step try olive.field("compile_olive").call(&vm, &.{
            try Value.new_string(&heap, olive_code),
        });
    };
    const olive_self_hosted_2 = step: {
        var step = try BootstrapStep.start(io, "Self-compiling Olive.", &heap);
        defer step.end(io);
        break :step try olive_self_hosted.field("compile_olive").call(&vm, &.{
            try Value.new_string(&heap, olive_code),
        });
    };
    _ = {
        var step = try BootstrapStep.start(io, "Confirming self-hosting.", &heap);
        defer step.end(io);
        if (!is_same(olive_self_hosted.obj, olive_self_hosted_2.obj)) {
            @panic("Not the same.");
        }
    };
    const compile_pear = step: {
        var step = try BootstrapStep.start(io, "Keeping only Pear compiler.", &heap);
        defer step.end(io);
        break :step Value.from(try vm.garbage_collect(start_of_heap, olive_self_hosted_2.field("compile_pear").obj));
    };

    std.debug.print(io, "Running notebook.\n", .{});
    const compiled = try compile_pear.call(&vm, &.{try Value.new_string(&heap, pear_code)});
    const make_app = try compiled.call(&vm, &.{});
    const book = Value.from(try object_loader.load(
        ally,
        &heap,
        try Io.Dir.cwd().readFileAlloc(io, "src/data.objects", ally, Io.Limit.unlimited),
    ));
    const app_val = try make_app.call(&vm, &.{book});
    const app = Value.from(try vm.garbage_collect(start_of_heap, app_val.obj));

    try App.run(ally, io, &heap, &vm, app, "src/data.objects");
}
