// This module JIT-compiles instructions to x86 machine code. The whole idea is
// to map the semantics of my VM to semantics of the x86-64 machine because it
// has a more efficient interpreter (the CPU). This is how I do it:
//
// 0. I create the VM (including memory for its stack and the heap) before doing
//    any machine code shenanigans. In particular, I allocate memory:
//
//    Heap:       The VM's heap is a memory region that I allocate before (the
//                Zig Heap struct does that). The heap can't grow dynamically;
//                if you use it too much, you get an out-of-memory error
//                instead.
//    Data stack: The VM's data stack is a memory region that I allocate before.
//                The data stack can't grow dynamically; code can overflow it.
//    Call stack: The VM's call stack is a memory region that we allocate before.
//                The call stack can't grow dynamically; code can overflow it.
//
// 1. Before running machine code, I prepare data structures to pass to it:
//
//    VM state:  The VM and heap contain Zig data structures (slices, etc.),
//               which don't have a defined memory layout. I extract relevant
//               parts (stack and heap cursors) into a VmState struct.
//    Zig state: I want to be able to call certain Zig code from the VM, for
//               example, to do a garbage collection without re-implementing
//               that in assembly. The Zig state bundles up stuff that this Zig
//               code needs (Allocator, Vm). I pass a pointer to the Zig state
//               to the machine code, which treats it like an opaque pointer.
//
// 2. I call a generated function with the C calling convention with the
//    prepared state. The machine code of my compiled functions doesn't use the
//    SysV calling convention (the C ABI), but a custom one instead:
//
//    rax = intra-instruction temporary state (e.g. idiv always uses rax)
//    rbx = intra-instruction temporary state
//    rcx = sandbox stack cursor
//    rdx = original rsp
//    rsi = fuel left
//    rdi = heap cursor
//    rsp = data stack cursor
//    rbp = call stack cursor
//    r8  = general purpose register
//    r9  = general purpose register
//    r10 = general purpose register
//    r11 = general purpose register
//    r12 = general purpose register
//    r13 = general purpose register
//    r14 = general purpose register
//    r15 = general purpose register
//
//    When you call a function, upon return, the following registers are
//    guaranteed to be unchanged: rdx, rsi, rdi, rsp, rbp.
//    All other registers may be overwritten.
//
//    To bridge the calling convention, I backup the SysV state (callee-saved
//    registers) on the x86 stack. I store the VM state and the Zig state on the
//    x86 stack. I copy heap cursor, stack cursors, and fuel into registers.
//    Then I store the original rsp in rdx and store the data stack cursor into
//    rsp. From that point, I can use push and pop instructions to move contents
//    between my stack and the registers. However, calls become more complicated
//    (I can't use the call instruction, since it would modify the data stack).
//
// 3. Generated machine code runs.
//
// 4. Upon returning to Zig code, I use the original rsp (backupped in rdx) to
//    get access to the Vm state and Zig state. I copy the VM's state from
//    registers into the Vm state and go back to Zig code.

const std = @import("std");
const ArrayList = std.ArrayList;
const Map = std.AutoArrayHashMap;
const Ally = std.mem.Allocator;

const Heap = @import("heap.zig");
const Word = Heap.Word;
const Obj = Heap.Obj;
const get_symbol = Heap.get_symbol;
const ObjMap = @import("obj_map.zig").ObjMap;
const Vm = @import("vm.zig");

const Self = @This();

ally: Ally,
// TODO: add guard pages next to the heap and stacks

heap: *Heap,
// The generated machine code will create objects with the exact same memory
// layout as Zig code or other VMs.

data_stack: Stack,
// The stack is private to this VM. Unlike the byte code interpreter, this stack
// grows from the top down (from high addresses toward low addresses; filling
// the allocated memory region from the back). The native x86 stack has the same
// property and it allows us to emit "push" and "pop" instructions that directly
// operate on the data stack.

call_stack: Stack,
// Also grows downward, contains callees.

sandbox_stack: Stack,
// Also grows downward. Every sandbox scope pushes a 6-word struct:
// { on_crash, heap, call, data, fuel_diff, on_out_of_fuel }. The translator
// from SysV to my calling convention adds a sandbox scope, so when a crash or
// out-of-fuel happens, there is guaranteed to be an entry on the stack.
// fuel_diff is how much fuel the sandbox subtracted from rsi when it capped
// the body's allowance, so it can be added back when the sandbox exits. The
// outer translator entry uses fuel_diff = 1 as a sentinel so the out-of-fuel
// walking loop (which skips entries with fuel_diff == 0) terminates at the
// translator instead of running off the bottom of the sandbox stack.

fuel: usize = 0,
// Remaining fuel for the currently-running call. Like the stacks, this only
// has a meaningful value during a call; `call` sets it from the caller's
// pointer and writes the leftover back when the call returns.

jit_cache: ObjMap([]const u8),
// Maps (pointer to instructions obj) to (x86_64 machine code). The machine code
// memory regions are marked as executable memory pages.

run_jitted_wrapper: *const fn (*VmState, *ZigState, [*]const u8) callconv(.c) usize,
// Returns a status code: 0 = returned normally, 1 = crashed.

// Laid out in C order so the JIT can index into it with fixed byte offsets.
const VmState = extern struct {
    heap_cursor: usize, // +0
    data_stack_cursor: usize, // +8
    call_stack_cursor: usize, // +16
    sandbox_stack_cursor: usize, // +24
    fuel: usize, // +32

    fn load_from_vm(self: *VmState, vm: *Self) void {
        self.heap_cursor = @intFromPtr(vm.heap.memory.ptr) + (8 * vm.heap.used);
        self.data_stack_cursor = @intFromPtr(vm.data_stack.memory.ptr) + (8 * vm.data_stack.cursor);
        self.call_stack_cursor = @intFromPtr(vm.call_stack.memory.ptr) + (8 * vm.call_stack.cursor);
        self.sandbox_stack_cursor = @intFromPtr(vm.sandbox_stack.memory.ptr) + (8 * vm.sandbox_stack.cursor);
        self.fuel = vm.fuel;
    }
    fn store_to_vm(self: *VmState, vm: *Self) void {
        vm.heap.used = (self.heap_cursor - @intFromPtr(vm.heap.memory.ptr)) / 8;
        vm.data_stack.cursor = (self.data_stack_cursor - @intFromPtr(vm.data_stack.memory.ptr)) / 8;
        vm.call_stack.cursor = (self.call_stack_cursor - @intFromPtr(vm.call_stack.memory.ptr)) / 8;
        vm.sandbox_stack.cursor = (self.sandbox_stack_cursor - @intFromPtr(vm.sandbox_stack.memory.ptr)) / 8;
        vm.fuel = self.fuel;
    }
};
const ZigState = struct {
    vm: *Self,
};

// The stack grows downward, just like the native x86 stack. This allows us to use push and pop
// instructions directly to modify the (data) stack.
const Stack = struct {
    memory: []Word,
    cursor: usize,

    pub fn init(ally: Ally, num_words: usize) !Stack {
        return .{ .memory = try ally.alloc(Word, num_words), .cursor = num_words };
    }
    pub fn push(self: *Stack, word: Word) !void {
        if (self.cursor == 0) @panic("stack overflow");
        self.cursor -= 1;
        self.memory[self.cursor] = word;
    }
    pub fn pop(self: *Stack) Word {
        const word = self.memory[self.cursor];
        self.cursor += 1;
        return word;
    }
};

pub fn init(heap: *Heap, ally: Ally) !Self {
    return init_sized(heap, ally, 200000000, 200000000);
}
// Like `init`, but with explicit data/call stack sizes (in words). Worker
// threads use small stacks so that spawning a VM per task is affordable; the
// default `init` keeps the large sizes the main VM relies on.
pub fn init_sized(heap: *Heap, ally: Ally, data_stack_words: usize, call_stack_words: usize) !Self {
    return .{
        .ally = ally,
        .heap = heap,
        .data_stack = try Stack.init(ally, data_stack_words),
        .call_stack = try Stack.init(ally, call_stack_words),
        .sandbox_stack = try Stack.init(ally, 1000),
        .jit_cache = ObjMap([]const u8).empty,
        .run_jitted_wrapper = wrapper: {
            // SysV args: rdi=*VmState, rsi=*ZigState, rdx=jitted machine code ptr.
            //
            // Native-stack layout after the prologue (low to high):
            //   [rdx + 0]  = *ZigState
            //   [rdx + 8]  = *VmState
            //   [...]      = saved callee-saved regs
            //
            // Returns (in rax): 0 = function returned normally, 1 = a .crash
            // unwound to our outer sandbox handler.
            var mc = MachineCode{ .ally = ally };
            defer mc.deinit();
            const sink = &mc;

            // Save SysV callee-saved registers we're about to clobber.
            try sink.emit("push {reg}", .{.rbx});
            try sink.emit("push {reg}", .{.rbp});
            try sink.emit("push {reg}", .{.r12});
            try sink.emit("push {reg}", .{.r13});
            try sink.emit("push {reg}", .{.r14});
            try sink.emit("push {reg}", .{.r15});

            // Park *VmState and *ZigState on the native stack so hooks can reach
            // them via [rdx + 8] / [rdx].
            try sink.emit("push {reg}", .{.rdi}); // *VmState
            try sink.emit("push {reg}", .{.rsi}); // *ZigState

            // Stash jitted_ptr (currently rdx) in rax so we can jmp to it later;
            // rdx is about to become the original-rsp marker.
            try sink.emit("mov {reg}, {reg}", .{ .rax, .rdx });

            // Switch from SysV to Orchard convention.
            try sink.emit("mov {reg}, {reg}", .{ .rdx, .rsp }); // rdx = original rsp
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 }); // rbx = *VmState
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rdi, .rbx, 0 }); // rdi = heap
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rbx, 16 }); // rbp = call_stack
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rcx, .rbx, 24 }); // rcx = sandbox_stack
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rbx, 32 }); // rsi = fuel
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rbx, 8 }); // rsp = data (LAST)

            // Install the outer sandbox entry. Its crash_target/ouf_target are
            // the labels below; on unwind, the jitted code restores cursors
            // from the entry and jumps there, where we'll set rax=1 or rax=2
            // and return. fuel_diff = 1 is a sentinel: the outer entry doesn't
            // actually cap fuel, but the out-of-fuel walking loop skips
            // entries with fuel_diff == 0, so we need a non-zero value to
            // ensure the loop terminates at the translator. The +1 that .crash
            // adds to rsi is harmless because the epilogue ignores rsi.
            try sink.emit("sub {reg}, {i32}", .{ .rcx, 48 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 24, .rsp }); // data snapshot
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 16, .rbp }); // call snapshot
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 8, .rdi }); // heap snapshot
            const crash_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 0, .rbx }); // crash_target
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 32, .rbx }); // fuel_diff sentinel
            const ouf_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 40, .rbx }); // on_out_of_fuel

            // "Call" the jitted code: push our return address onto the Orchard
            // call stack and jmp into it.
            try sink.emit("sub {reg}, {i32}", .{ .rbp, 8 });
            const ret_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbp, 0, .rbx });
            try sink.emit("jmp {reg}", .{.rax});

            // --- Normal-return path ---
            try sink.patch_to_here(ret_patch);
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop our outer sandbox entry
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 0) }); // status = returned
            const to_epilogue_from_normal = try sink.placeholder("jmp {label}", .{});

            // --- Crash path ---
            // .crash already restored cursors and popped its sandbox entry.
            // It also added our entry's fuel_diff sentinel (1) to rsi to
            // "restore" the outer slack — subtract it back here so the user
            // sees the actual remaining fuel at crash time.
            try sink.patch_to_here(crash_patch);
            try sink.emit("sub {reg}, {i32}", .{ .rsi, 1 });
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 1) }); // status = crashed
            const to_epilogue_from_crash = try sink.placeholder("jmp {label}", .{});

            // --- Out-of-fuel path ---
            // The .use_fuel unwinder already restored cursors and popped its
            // sandbox entry, and set rsi to our entry's fuel_diff (1, sentinel
            // garbage). Force rsi = 0 to match the byte-code's contract for
            // uncaught out-of-fuel (`self.fuel = 0`).
            try sink.patch_to_here(ouf_patch);
            try sink.emit("xor {reg}, {reg}", .{ .rsi, .rsi });
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 2) }); // status = out_of_fuel
            try sink.patch_to_here(to_epilogue_from_normal);
            try sink.patch_to_here(to_epilogue_from_crash);

            // --- Shared epilogue ---
            // Write the updated cursors and remaining fuel back into *VmState.
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 }); // rbx = *VmState
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 0, .rdi });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 8, .rsp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 16, .rbp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 24, .rcx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 32, .rsi });

            // Switch back to the native stack and discard the parked pointers.
            try sink.emit("mov {reg}, {reg}", .{ .rsp, .rdx });
            try sink.emit("pop {reg}", .{.rsi}); // discard *ZigState
            try sink.emit("pop {reg}", .{.rdi}); // discard *VmState

            // Restore SysV callee-saved registers, in reverse order. rax (the
            // status code) is preserved across all these.
            try sink.emit("pop {reg}", .{.r15});
            try sink.emit("pop {reg}", .{.r14});
            try sink.emit("pop {reg}", .{.r13});
            try sink.emit("pop {reg}", .{.r12});
            try sink.emit("pop {reg}", .{.rbp});
            try sink.emit("pop {reg}", .{.rbx});

            try sink.emit("ret", .{});

            break :wrapper @ptrCast(try sink.finalize());
        },
    };
}
pub fn deinit(_: *Self) !void {
    // TODO: std.linux.munmap() the cache
}

pub fn call(vm: *Self, fun: Vm.Fun, args: []const Obj, fuel: *usize) !Vm.Result {
    vm.data_stack.cursor = vm.data_stack.memory.len;
    vm.call_stack.cursor = vm.call_stack.memory.len;
    vm.sandbox_stack.cursor = vm.sandbox_stack.memory.len;
    vm.fuel = fuel.*;
    defer fuel.* = vm.fuel;

    const jitted = vm.compile(fun) catch |err| switch (err) {
        error.OutOfMemory => return .out_of_memory,
        error.UndefinedBehavior => return error.UndefinedBehavior,
    };
    for (args) |arg| vm.data_stack.push(arg.address) catch return .out_of_memory;
    const status = vm.run_jitted(jitted) catch |err| switch (err) {
        error.OutOfMemory => return .out_of_memory,
    };
    return switch (status) {
        0 => .{ .returned = .{ .address = vm.data_stack.pop() } },
        1 => .{ .crashed = .{ .address = vm.data_stack.pop() } },
        2 => .out_of_fuel,
        else => unreachable,
    };
}
pub fn run_jitted(vm: *Self, jitted: []const u8) error{OutOfMemory}!usize {
    // std.debug.print("data stack:", .{});
    // for (vm.data_stack.memory[vm.data_stack.cursor..]) |word| std.debug.print(" {x:016}", .{word});
    // std.debug.print("\n", .{});
    // std.debug.print("running jitted instructions\n", .{});

    var zig_state = ZigState{ .vm = vm };
    var vm_state: VmState = undefined;
    vm_state.load_from_vm(vm);
    const status = vm.run_jitted_wrapper(&vm_state, &zig_state, @ptrCast(jitted));
    vm_state.store_to_vm(vm);

    // std.debug.print("ran jitted instructions (status={d})\n", .{status});
    // std.debug.print("data stack:", .{});
    // for (vm.data_stack.memory[vm.data_stack.cursor..]) |word| std.debug.print(" {x:016}", .{word});
    // std.debug.print("\n", .{});
    return status;
}
pub fn compile(vm: *Self, fun: Vm.Fun) !([]const u8) {
    if (vm.jit_cache.get(fun.obj)) |machine_code| return machine_code;
    const machine_code = try compile_fun(vm, fun);
    try vm.jit_cache.put(vm.ally, fun.obj, machine_code);
    return machine_code;
}

// Compiles a Vm.Fun directly to x86 machine code. We track named values on a
// parallel "name stack" so .name lookups become `push [rsp + offset*8]`.
pub fn compile_fun(vm: *Self, fun: Vm.Fun) !([]const u8) {
    const ally = vm.ally;

    var arena = std.heap.ArenaAllocator.init(ally);
    defer arena.deinit();
    const ir = parse_ir(vm, arena.allocator(), fun) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.UndefinedBehavior => return error.UndefinedBehavior,
    };
    compute_regs(ir);
    // std.debug.print("== fun ==\n{f}\n", .{fun});
    // std.debug.print("== jit ir ==\n{f}", .{ir});

    // var asm_sink = try Asm.init(arena.allocator());
    // defer asm_sink.deinit();
    // try emit_fun(&asm_sink, ir, fun);
    // std.debug.print("== asm ==\n{s}", .{asm_sink.bytes.items});

    var mc = MachineCode{ .ally = ally };
    defer mc.deinit();
    try emit_fun(&mc, ir, fun);
    // std.debug.print("== machine code ==\n", .{});
    // for (mc.bytes.items) |b| std.debug.print(" {x:02}", .{b});
    // std.debug.print("\n", .{});

    return try mc.finalize();
}

// JIT IR: Similar to the Zig compiler, I turn expressions into a prefix-encoded
// dense representation with variable-sized nodes. Each expression occupies a
// contiguous run of slots that start with a header. Children come afterwards.
//
// Example: (add (word 2) (word 3))
// Becomes: [add ][word][   2][word][   3]
//
// This means that I can't just go to e.g. the second operand of an add node,
// but I have to traverse nodes depth-first from the start.

pub const Ir = struct {
    slots: []u64,

    pub fn format(ir: Ir, writer: *std.Io.Writer) error{WriteFailed}!void {
        var cursor: Index = 0;
        _ = try dump_node(ir.slots, &cursor, 0, writer);
    }
};
const Index = usize;
const Header = packed struct {
    tag: Tag,
    regs: u8 = 0, // filled in later by the regs pass
    rest: u48 = 0, // tag-specific payload
};
const Tag = enum(u8) {
    word, // value (u64)
    object, // address (pointer)
    name, // name (pointer into VM heap's symbol object)
    let, // name (pointer into VM heap's symbol object), def, body
    also, // ignored, value
    add, // left, right
    subtract, // left, right
    multiply, // left, right
    divide, // left, right
    modulo, // left, right
    shift_left, // left, right
    shift_right, // left, right
    and_, // left, right
    or_, // left, right
    xor, // left, right
    compare, // left, right
    f_add, // left, right
    f_subtract, // left, right
    f_multiply, // left, right
    f_divide, // left, right
    f_compare, // left, right
    int_to_float, // child
    float_to_int, // child
    f_is_finite, // child
    if_, // cond, then, else
    new_leaf, // [rest = num children] child 0, child 1, ...
    new_inner, // [rest = num children] child 0, child 1, ...
    flatten_to_leaf, // child
    flatten_to_inner, // child
    points, // child
    size, // child
    load, // object, index
    load_imm, // index (usize), object
    call, // [rest = num args] arg 0, arg1, ..., fun
    call_imm, // [rest = num args] fun (heap obj), arg 0, arg 1, ...
    call_indirect, // [rest = num args] args, fun
    rec, // [rest = num args] arg 0, arg 1, ...
    collect_garbage, // child
    sandbox, // fuel, body
    use_fuel, // amount, child
    crash, // error
    unreachable_, // header
};

fn eat_slot(slots: []const u64, at: *Index) u64 {
    const slot = slots[at.*];
    at.* += 1;
    return slot;
}
fn dump_node(slots: []const u64, at: *Index, indent: usize, writer: *std.Io.Writer) error{WriteFailed}!void {
    for (0..indent) |_| try writer.print("  ", .{});
    const header: Header = @bitCast(eat_slot(slots, at));
    try writer.print("{s}", .{@tagName(header.tag)});
    switch (header.tag) {
        .word => try writer.print(" {d}\n", .{eat_slot(slots, at)}),
        .object => try writer.print(" 0x{x}\n", .{eat_slot(slots, at)}),
        .name => {
            const name_obj = Obj{ .address = eat_slot(slots, at) };
            try writer.print(" ", .{});
            try Obj.format_string(get_symbol(name_obj), false, writer);
            try writer.print("\n", .{});
        },
        .let => {
            const name_obj = Obj{ .address = eat_slot(slots, at) };
            try writer.print(" ", .{});
            try Obj.format_string(get_symbol(name_obj), false, writer);
            try writer.print("\n", .{});
            try dump_node(slots, at, indent + 1, writer); // def
            try dump_node(slots, at, indent + 1, writer); // body
        },
        .also,
        .add,
        .subtract,
        .multiply,
        .divide,
        .modulo,
        .shift_left,
        .shift_right,
        .and_,
        .or_,
        .xor,
        .compare,
        .f_add,
        .f_subtract,
        .f_multiply,
        .f_divide,
        .f_compare,
        .load,
        .call_indirect,
        .sandbox,
        .use_fuel,
        => {
            try writer.print("\n", .{});
            try dump_node(slots, at, indent + 1, writer);
            try dump_node(slots, at, indent + 1, writer);
        },
        .int_to_float,
        .float_to_int,
        .f_is_finite,
        .flatten_to_leaf,
        .flatten_to_inner,
        .points,
        .size,
        .crash,
        .collect_garbage,
        => {
            try writer.print("\n", .{});
            try dump_node(slots, at, indent + 1, writer);
        },
        .if_ => {
            try writer.print("\n", .{});
            try dump_node(slots, at, indent + 1, writer); // cond
            try dump_node(slots, at, indent + 1, writer); // then
            try dump_node(slots, at, indent + 1, writer); // else
        },
        .new_leaf, .new_inner, .rec => {
            const n: usize = @intCast(header.rest);
            try writer.print("\n", .{});
            for (0..n) |_| try dump_node(slots, at, indent + 1, writer);
        },
        .call => {
            const n: usize = @intCast(header.rest);
            try writer.print("\n", .{});
            for (0..n) |_| try dump_node(slots, at, indent + 1, writer); // args
            try dump_node(slots, at, indent + 1, writer); // fun
        },
        .call_imm => {
            const n: usize = @intCast(header.rest);
            try writer.print(" 0x{x}\n", .{eat_slot(slots, at)});
            for (0..n) |_| try dump_node(slots, at, indent + 1, writer);
        },
        .load_imm => {
            try writer.print(" {d}\n", .{eat_slot(slots, at)});
            try dump_node(slots, at, indent + 1, writer);
        },
        .unreachable_ => try writer.print("\n", .{}),
    }
}

pub fn parse_ir(vm: *Self, ally: Ally, fun: Vm.Fun) !Ir {
    var slots = ArrayList(u64).empty;
    errdefer slots.deinit(ally);
    try parse_node(vm, ally, &slots, fun.body());
    return .{ .slots = try slots.toOwnedSlice(ally) };
}
fn parse_node(vm: *Self, ally: Ally, slots: *ArrayList(u64), expr: Vm.Expr) error{ OutOfMemory, UndefinedBehavior }!void {
    switch (try expr.kind()) {
        .word => |word| {
            try slots.append(ally, @bitCast(Header{ .tag = .word }));
            try slots.append(ally, word);
        },
        .object => |obj| {
            try slots.append(ally, @bitCast(Header{ .tag = .object }));
            try slots.append(ally, obj.address);
        },
        .name => {
            try slots.append(ally, @bitCast(Header{ .tag = .name }));
            try slots.append(ally, @bitCast(expr.obj.child(1).address));
        },
        .let => |let| {
            try slots.append(ally, @bitCast(Header{ .tag = .let }));
            try slots.append(ally, @bitCast(expr.obj.child(1).address));
            try parse_node(vm, ally, slots, let.def);
            try parse_node(vm, ally, slots, let.body);
        },
        .also => |also_| {
            try slots.append(ally, @bitCast(Header{ .tag = .also }));
            try parse_node(vm, ally, slots, also_.ignored);
            try parse_node(vm, ally, slots, also_.value);
        },
        .add => |args| try parse_binop(vm, ally, slots, .add, args),
        .subtract => |args| try parse_binop(vm, ally, slots, .subtract, args),
        .multiply => |args| try parse_binop(vm, ally, slots, .multiply, args),
        .divide => |args| try parse_binop(vm, ally, slots, .divide, args),
        .modulo => |args| try parse_binop(vm, ally, slots, .modulo, args),
        .shift_left => |args| try parse_binop(vm, ally, slots, .shift_left, args),
        .shift_right => |args| try parse_binop(vm, ally, slots, .shift_right, args),
        .and_ => |args| try parse_binop(vm, ally, slots, .and_, args),
        .or_ => |args| try parse_binop(vm, ally, slots, .or_, args),
        .xor => |args| try parse_binop(vm, ally, slots, .xor, args),
        .compare => |args| try parse_binop(vm, ally, slots, .compare, args),
        .f_add => |args| try parse_binop(vm, ally, slots, .f_add, args),
        .f_subtract => |args| try parse_binop(vm, ally, slots, .f_subtract, args),
        .f_multiply => |args| try parse_binop(vm, ally, slots, .f_multiply, args),
        .f_divide => |args| try parse_binop(vm, ally, slots, .f_divide, args),
        .f_compare => |args| try parse_binop(vm, ally, slots, .f_compare, args),
        .int_to_float => |arg| try parse_unop(vm, ally, slots, .int_to_float, arg),
        .float_to_int => |arg| try parse_unop(vm, ally, slots, .float_to_int, arg),
        .f_is_finite => |arg| try parse_unop(vm, ally, slots, .f_is_finite, arg),
        .flatten_to_leaf => |arg| try parse_unop(vm, ally, slots, .flatten_to_leaf, arg),
        .flatten_to_inner => |arg| try parse_unop(vm, ally, slots, .flatten_to_inner, arg),
        .points => |arg| try parse_unop(vm, ally, slots, .points, arg),
        .size => |arg| try parse_unop(vm, ally, slots, .size, arg),
        .crash => |arg| try parse_unop(vm, ally, slots, .crash, arg),
        .collect_garbage => |arg| try parse_unop(vm, ally, slots, .collect_garbage, arg),
        .if_ => |if_| {
            try slots.append(ally, @bitCast(Header{ .tag = .if_ }));
            try parse_node(vm, ally, slots, if_.condition);
            try parse_node(vm, ally, slots, if_.then);
            try parse_node(vm, ally, slots, if_.else_);
        },
        .new_leaf => |children| {
            try slots.append(ally, @bitCast(Header{ .tag = .new_leaf, .rest = @intCast(children.len) }));
            for (children) |child| try parse_node(vm, ally, slots, child);
        },
        .new_inner => |children| {
            try slots.append(ally, @bitCast(Header{ .tag = .new_inner, .rest = @intCast(children.len) }));
            for (children) |child| try parse_node(vm, ally, slots, child);
        },
        .load => |load| {
            const index_kind = try load.index.kind();
            if (index_kind == .word) {
                try slots.append(ally, @bitCast(Header{ .tag = .load_imm }));
                try slots.append(ally, index_kind.word);
                try parse_node(vm, ally, slots, load.object);
            } else {
                try slots.append(ally, @bitCast(Header{ .tag = .load }));
                try parse_node(vm, ally, slots, load.object);
                try parse_node(vm, ally, slots, load.index);
            }
        },
        .call => |c| {
            const fun_kind = try c.fun.kind();
            if (fun_kind == .object) {
                const machine_code = try vm.compile(.{ .obj = fun_kind.object });
                try slots.append(ally, @bitCast(Header{ .tag = .call_imm, .rest = @intCast(c.args.len) }));
                try slots.append(ally, @intFromPtr(machine_code.ptr));
                for (c.args) |arg| try parse_node(vm, ally, slots, arg);
                return;
            }
            try slots.append(ally, @bitCast(Header{ .tag = .call, .rest = @intCast(c.args.len) }));
            for (c.args) |arg| try parse_node(vm, ally, slots, arg);
            try parse_node(vm, ally, slots, c.fun);
        },
        .call_indirect => |c| {
            try slots.append(ally, @bitCast(Header{ .tag = .call_indirect }));
            try parse_node(vm, ally, slots, c.args);
            try parse_node(vm, ally, slots, c.fun);
        },
        .rec => |args| {
            try slots.append(ally, @bitCast(Header{ .tag = .rec, .rest = @intCast(args.len) }));
            for (args) |arg| try parse_node(vm, ally, slots, arg);
        },
        .sandbox => |s| {
            try slots.append(ally, @bitCast(Header{ .tag = .sandbox }));
            try parse_node(vm, ally, slots, s.fuel);
            try parse_node(vm, ally, slots, s.body);
        },
        .use_fuel => |uf| {
            try slots.append(ally, @bitCast(Header{ .tag = .use_fuel }));
            try parse_node(vm, ally, slots, uf.amount);
            try parse_node(vm, ally, slots, uf.child);
        },
        .unreachable_ => {
            try slots.append(ally, @bitCast(Header{ .tag = .unreachable_ }));
        },
    }
}
fn parse_binop(vm: *Self, ally: Ally, slots: *ArrayList(u64), tag: Tag, args: Vm.Expr.Kind.LeftRight) !void {
    try slots.append(ally, @bitCast(Header{ .tag = tag }));
    try parse_node(vm, ally, slots, args.left);
    try parse_node(vm, ally, slots, args.right);
}
fn parse_unop(vm: *Self, ally: Ally, slots: *ArrayList(u64), tag: Tag, arg: Vm.Expr) !void {
    try slots.append(ally, @bitCast(Header{ .tag = tag }));
    try parse_node(vm, ally, slots, arg);
}

// Computes how many registers each subtrees needs and stores it directly into
// the node's header. If a subtree can't be evaluated only using registers
// (e.g. because it calls into Zig code, like garbage_collect), it stores STACK
// instead.
const STACK: u8 = 255;
pub fn compute_regs(ir: Ir) void {
    var cursor: Index = 0;
    _ = compute_node_regs(ir.slots, &cursor);
}
fn also(_: anytype, value: anytype) @TypeOf(value) {
    return value;
}
fn compute_node_regs(slots: []u64, at: *Index) u8 {
    const header_idx = at.*;
    var header: Header = @bitCast(eat_slot(slots, at));

    const regs: u8 = switch (header.tag) {
        .word => also(eat_slot(slots, at), @as(u8, 1)),
        .object => also(eat_slot(slots, at), @as(u8, 1)),
        .name => also(eat_slot(slots, at), @as(u8, 1)),
        .let => also(
            eat_slot(slots, at),
            @max(
                compute_node_regs(slots, at),
                1 +| compute_node_regs(slots, at),
            ),
        ),
        .also => @max(
            compute_node_regs(slots, at),
            compute_node_regs(slots, at),
        ),
        .add, .subtract, .multiply, .divide, .modulo, .shift_left, .shift_right, .and_, .or_, .xor, .compare => @max(
            compute_node_regs(slots, at),
            1 +| compute_node_regs(slots, at),
        ),
        .f_add, .f_subtract, .f_multiply, .f_divide, .f_compare => @max(
            compute_node_regs(slots, at),
            1 +| compute_node_regs(slots, at),
        ),
        .int_to_float, .float_to_int, .f_is_finite => compute_node_regs(slots, at),
        .if_ => @max(
            compute_node_regs(slots, at),
            compute_node_regs(slots, at),
            compute_node_regs(slots, at),
        ),
        .new_leaf, .new_inner => blk: {
            const n: usize = @intCast(header.rest);
            if (n > 255) {
                for (0..n) |_| _ = compute_node_regs(slots, at);
                break :blk STACK;
            } else {
                var max: u8 = 1;
                for (0..n) |i|
                    max = @max(max, @as(u8, @intCast(i)) +| compute_node_regs(slots, at));
                break :blk max;
            }
        },
        .flatten_to_leaf, .flatten_to_inner => also(
            compute_node_regs(slots, at),
            STACK,
        ),
        .points, .size => compute_node_regs(slots, at),
        .load => @max(
            compute_node_regs(slots, at),
            1 +| compute_node_regs(slots, at),
        ),
        .load_imm => also(
            eat_slot(slots, at),
            compute_node_regs(slots, at),
        ),
        .call => blk: {
            const n: usize = @intCast(header.rest);
            for (0..n) |_| _ = compute_node_regs(slots, at);
            _ = compute_node_regs(slots, at); // fun
            break :blk STACK;
        },
        .call_imm => blk: {
            _ = eat_slot(slots, at);
            const n: usize = @intCast(header.rest);
            for (0..n) |_| _ = compute_node_regs(slots, at);
            break :blk STACK;
        },
        .call_indirect => blk: {
            _ = compute_node_regs(slots, at);
            _ = compute_node_regs(slots, at);
            break :blk STACK;
        },
        .rec => blk: {
            const n: usize = @intCast(header.rest);
            for (0..n) |_| _ = compute_node_regs(slots, at);
            break :blk STACK;
        },
        .sandbox => @max(
            compute_node_regs(slots, at),
            compute_node_regs(slots, at),
            STACK,
        ),
        .use_fuel => blk: {
            _ = compute_node_regs(slots, at);
            _ = compute_node_regs(slots, at);
            break :blk STACK;
        },
        .collect_garbage => also(
            compute_node_regs(slots, at),
            STACK,
        ),
        .crash => compute_node_regs(slots, at),
        .unreachable_ => 1,
    };
    header.regs = regs;
    slots[header_idx] = @bitCast(header);
    return regs;
}

// Next, I define helpers for creating assembly or machine code. Both are
// implemented as structs with an emit function that you can call with a
// comptime assembly string.
//
// Functions that emit something use an anytype "sink", which will either be:
//
// - Asm: A sink that generates Intel-syntax assembly text with comments and
//   indentation, useful for debugging.
// - Machine code: A sink that generates raw x86-64 machine code.
//
// Forward jumps emit placeholder bytes and return a Patch. patch_to_here later
// writes the relative offset for MachineCode or appends a label line for Asm.
// Patches must be patched in the same emitter they came from.

const Reg = enum(u4) {
    rax = 0,
    rcx = 1,
    rdx = 2,
    rbx = 3,
    rsp = 4,
    rbp = 5,
    rsi = 6,
    rdi = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,

    fn next(self: Reg) Reg {
        return self.add(1);
    }
    fn add(self: Reg, i: usize) Reg {
        return @enumFromInt(@intFromEnum(self) + i);
    }
};

const Patch = struct {
    // Asm: id of the label we'll define later via patch_to_here.
    // MachineCode: byte offset of the i32 displacement slot.
    handle: u32,
};

// Marks an already-emitted location that later instructions can jump back to.
// Asm: id of a label written at the target site.
// MachineCode: absolute byte offset of the target inside the buffer.
const BackLabel = struct { handle: u32 };

const Asm = struct {
    ally: Ally,
    bytes: ArrayList(u8) = .empty,
    next_label: u32 = 0,
    indentation: usize = 0,

    fn init(ally: Ally) !Asm {
        var self = Asm{ .ally = ally };
        try self.write("root:\n", .{});
        return self;
    }
    fn deinit(self: *Asm) void {
        self.bytes.deinit(self.ally);
    }

    fn write(self: *Asm, comptime fmt: []const u8, args: anytype) !void {
        var buf: [256]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch @panic("asm line too long");
        try self.bytes.appendSlice(self.ally, s);
    }

    fn indent(self: *Asm) void {
        self.indentation += 1;
    }
    fn deindent(self: *Asm) void {
        self.indentation -= 1;
    }
    fn comment(self: *Asm, comptime fmt: []const u8, args: anytype) !void {
        for (0..self.indentation) |_| try self.write("  ", .{});
        try self.write("; ", .{});
        try self.write(fmt, args);
        try self.write("\n", .{});
    }
    fn emit(self: *Asm, comptime instr: []const u8, args: anytype) !void {
        for (0..self.indentation) |_| try self.write("  ", .{});
        if (comptime std.mem.eql(u8, instr, "jmp {root}")) {
            try self.write("jmp .root\n", .{});
            return;
        }
        const Token = struct {
            kind: enum { literal, placeholder },
            text: []const u8,
        };
        const tokens = comptime tokens: {
            var tokens: []const Token = &[_]Token{};
            var lit_start: usize = 0;
            var i: usize = 0;
            while (i < instr.len) {
                if (instr[i] == '{') {
                    if (i > lit_start) {
                        tokens = tokens ++ &[_]Token{.{ .kind = .literal, .text = instr[lit_start..i] }};
                    }
                    var end: usize = i + 1;
                    while (end < instr.len and instr[end] != '}') : (end += 1) {}
                    tokens = tokens ++ &[_]Token{.{ .kind = .placeholder, .text = instr[i + 1 .. end] }};
                    i = end + 1;
                    lit_start = i;
                } else {
                    i += 1;
                }
            }
            if (lit_start < instr.len) {
                tokens = tokens ++ &[_]Token{.{ .kind = .literal, .text = instr[lit_start..] }};
            }
            break :tokens tokens;
        };
        comptime var arg_idx: usize = 0;
        inline for (tokens) |tok| switch (tok.kind) {
            .literal => try self.write("{s}", .{tok.text}),
            .placeholder => {
                const arg = args[arg_idx];
                arg_idx += 1;
                if (comptime std.mem.eql(u8, tok.text, "reg")) {
                    try self.write("{s}", .{@tagName(arg)});
                } else if (comptime std.mem.eql(u8, tok.text, "reg8")) {
                    const low_name = ([_][]const u8{
                        "al",  "cl",  "dl",   "bl",   "spl",  "bpl",  "sil",  "dil",
                        "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
                    })[@intFromEnum(arg)];
                    try self.write("{s}", .{low_name});
                } else if (comptime std.mem.eql(u8, tok.text, "i32")) {
                    try self.write("{d}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "i8")) {
                    try self.write("{d}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "u8")) {
                    try self.write("{d}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "u64")) {
                    try self.write("0x{x}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "i64")) {
                    try self.write("{d}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "label")) {
                    try self.write(".L{d}", .{arg});
                } else if (comptime std.mem.eql(u8, tok.text, "back")) {
                    try self.write(".L{d}", .{arg.handle});
                } else {
                    @compileError("Asm.emit: unknown placeholder {" ++ tok.text ++ "}");
                }
            },
        };
        try self.write("\n", .{});
    }
    // Emits an instruction that contains a {label} placeholder referring to a
    // not-yet-known forward location. The returned Patch is later resolved
    // by patch_to_here at the actual target site.
    fn placeholder(self: *Asm, comptime instr: []const u8, args: anytype) !Patch {
        const id = self.next_label;
        self.next_label += 1;
        const expanded_args = args ++ .{id};
        if (comptime (std.mem.eql(u8, instr, "jz {label}") or
            std.mem.eql(u8, instr, "jb {label}") or
            std.mem.eql(u8, instr, "jp {label}") or
            std.mem.eql(u8, instr, "jmp {label}") or
            std.mem.eql(u8, instr, "lea {reg}, [rip + {label}]")))
        {
            try self.emit(instr, expanded_args);
            return .{ .handle = id };
        } else {
            @compileError("Asm.placeholder: unknown placeholder: " ++ instr);
        }
    }
    fn patch_to_here(self: *Asm, p: Patch) !void {
        for (0..self.indentation) |_| try self.write("  ", .{});
        try self.write(".L{d}:\n", .{p.handle});
    }
    // Defines a label at the current location and returns a BackLabel that
    // later "jmp {back}" instructions can target.
    fn here(self: *Asm) !BackLabel {
        const id = self.next_label;
        self.next_label += 1;
        for (0..self.indentation) |_| try self.write("  ", .{});
        try self.write(".L{d}:\n", .{id});
        return .{ .handle = id };
    }
};

const MachineCode = struct {
    ally: Ally,
    bytes: ArrayList(u8) = .empty,

    fn deinit(self: *MachineCode) void {
        self.bytes.deinit(self.ally);
    }

    // Copy the emitted bytes into a fresh mmap'd page with RX permissions,
    // so the CPU can actually execute them. The buffer (self.bytes) can be
    // freed by deinit afterwards — the returned slice is a separate region.
    fn finalize(self: *MachineCode) ![]const u8 {
        const len = self.bytes.items.len;
        if (len == 0) return &[_]u8{};
        const rw: std.os.linux.PROT = .{ .READ = true, .WRITE = true };
        const rx: std.os.linux.PROT = .{ .READ = true, .EXEC = true };
        const addr = std.os.linux.mmap(
            null,
            len,
            rw,
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        // mmap returns a small negative value (cast to usize: very large) on
        // failure.
        if (addr > std.math.maxInt(isize)) return error.OutOfMemory;
        const ptr: [*]u8 = @ptrFromInt(addr);
        @memcpy(ptr, self.bytes.items);
        std.debug.assert(std.os.linux.mprotect(ptr, len, rx) == 0);
        return ptr[0..len];
    }

    fn indent(_: *MachineCode) void {}
    fn deindent(_: *MachineCode) void {}
    fn comment(_: *MachineCode, _: []const u8, _: anytype) !void {}
    fn high(self: Reg) bool {
        return @intFromEnum(self) >= 8;
    }
    fn low(self: Reg) u3 {
        return @intCast(@intFromEnum(self) & 0x07);
    }
    fn b(self: *MachineCode, x: u8) !void {
        try self.bytes.append(self.ally, x);
    }
    fn i32le(self: *MachineCode, x: i32) !void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(i32, &buf, x, .little);
        try self.bytes.appendSlice(self.ally, &buf);
    }
    fn u64le(self: *MachineCode, x: u64) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, x, .little);
        try self.bytes.appendSlice(self.ally, &buf);
    }
    fn rex(self: *MachineCode, w: bool, r_ext: bool, x_ext: bool, b_ext: bool) !void {
        const byte: u8 = 0x40 |
            (@as(u8, @intFromBool(w)) << 3) |
            (@as(u8, @intFromBool(r_ext)) << 2) |
            (@as(u8, @intFromBool(x_ext)) << 1) |
            @intFromBool(b_ext);
        if (byte != 0x40) try self.b(byte);
    }
    fn modrm(self: *MachineCode, mod: u2, reg: u3, rm: u3) !void {
        try self.b((@as(u8, mod) << 6) | (@as(u8, reg) << 3) | @as(u8, rm));
    }
    fn mem_operand(self: *MachineCode, reg_field: u3, base: Reg, disp: i32) !void {
        const base_lo: u3 = low(base);
        const needs_sib = base_lo == 0b100;
        const is_rbp = base_lo == 0b101;
        const mod: u2 = blk: {
            if (disp == 0 and !is_rbp) break :blk 0b00;
            if (disp >= -128 and disp <= 127) break :blk 0b01;
            break :blk 0b10;
        };
        try self.modrm(mod, reg_field, base_lo);
        if (needs_sib) {
            try self.b(@as(u8, 0b00 << 6) | @as(u8, 0b100 << 3) | @as(u8, base_lo));
        }
        switch (mod) {
            0b00 => {},
            0b01 => try self.b(@as(u8, @bitCast(@as(i8, @intCast(disp))))),
            0b10 => try self.i32le(disp),
            else => unreachable,
        }
    }
    fn emit(self: *MachineCode, comptime instr: []const u8, args: anytype) !void {
        if (comptime std.mem.eql(u8, instr, "push {reg}")) {
            if (high(args[0])) try self.b(0x41);
            try self.b(0x50 | @as(u8, low(args[0])));
        } else if (comptime std.mem.eql(u8, instr, "pop {reg}")) {
            if (high(args[0])) try self.b(0x41);
            try self.b(0x58 | @as(u8, low(args[0])));
        } else if (comptime std.mem.eql(u8, instr, "push {i32}")) {
            try self.b(0x68);
            try self.i32le(args[0]);
        } else if (comptime std.mem.eql(u8, instr, "push {i8}")) {
            try self.b(0x6a);
            try self.b(@as(u8, @bitCast(@as(i8, args[0]))));
        } else if (comptime std.mem.eql(u8, instr, "push [{reg} + {i32}]")) {
            try self.rex(false, false, false, high(args[0]));
            try self.b(0xff);
            try self.mem_operand(6, args[0], args[1]);
        } else if (comptime std.mem.eql(u8, instr, "mov {reg}, {u64}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xb8 | @as(u8, low(args[0])));
            try self.u64le(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "mov {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x89);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "mov {reg}, [{reg} + {i32}]")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x8b);
            try self.mem_operand(low(args[0]), args[1], args[2]);
        } else if (comptime std.mem.eql(u8, instr, "mov [{reg} + {i32}], {reg}")) {
            try self.rex(true, high(args[2]), false, high(args[0]));
            try self.b(0x89);
            try self.mem_operand(low(args[2]), args[0], args[1]);
        } else if (comptime std.mem.eql(u8, instr, "add {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x01);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "sub {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x29);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "and {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x21);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "or {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x09);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "xor {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x31);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "imul {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0xaf);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "add {reg}, {i32}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x81);
            try self.modrm(0b11, 0, low(args[0]));
            try self.i32le(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "sub {reg}, {i32}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x81);
            try self.modrm(0b11, 5, low(args[0]));
            try self.i32le(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "cqo")) {
            try self.b(0x48);
            try self.b(0x99);
        } else if (comptime std.mem.eql(u8, instr, "idiv {reg}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xf7);
            try self.modrm(0b11, 7, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "shl {reg}, cl")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xd3);
            try self.modrm(0b11, 4, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "sar {reg}, cl")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xd3);
            try self.modrm(0b11, 7, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "shl {reg}, {u8}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xc1);
            try self.modrm(0b11, 4, low(args[0]));
            try self.b(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "shr {reg}, {u8}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xc1);
            try self.modrm(0b11, 5, low(args[0]));
            try self.b(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "sar {reg}, {u8}")) {
            try self.rex(true, false, false, high(args[0]));
            try self.b(0xc1);
            try self.modrm(0b11, 7, low(args[0]));
            try self.b(args[1]);
        } else if (comptime std.mem.eql(u8, instr, "shl ecx, 1")) {
            try self.b(0xd1);
            try self.modrm(0b11, 4, 1);
        } else if (comptime std.mem.eql(u8, instr, "cmp {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x39);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "test {reg}, {reg}")) {
            try self.rex(true, high(args[1]), false, high(args[0]));
            try self.b(0x85);
            try self.modrm(0b11, low(args[1]), low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "setz {reg8}")) {
            if (@intFromEnum(args[0]) >= 4) try self.rex(false, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x94);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "setnz {reg8}")) {
            if (@intFromEnum(args[0]) >= 4) try self.rex(false, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x95);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "setg {reg8}")) {
            if (@intFromEnum(args[0]) >= 4) try self.rex(false, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x9f);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "setl {reg8}")) {
            if (@intFromEnum(args[0]) >= 4) try self.rex(false, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x9c);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "setg ch")) {
            try self.b(0x0f);
            try self.b(0x9f);
            try self.modrm(0b11, 0, 5);
        } else if (comptime std.mem.eql(u8, instr, "lea {reg}, [rcx + rcx*2]")) {
            try self.rex(true, high(args[0]), false, false);
            try self.b(0x8d);
            try self.modrm(0b00, low(args[0]), 0b100); // SIB follows
            try self.b((@as(u8, 0b01) << 6) | (@as(u8, 0b001) << 3) | @as(u8, 0b001));
        } else if (comptime std.mem.eql(u8, instr, "cmovz {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0x44);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "cmovg {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0x4f);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "cmova {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0x47);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "cmovl {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0x4c);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "jmp {reg}")) {
            if (high(args[0])) try self.b(0x41);
            try self.b(0xff);
            try self.modrm(0b11, 4, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "call {reg}")) {
            if (high(args[0])) try self.b(0x41);
            try self.b(0xff);
            try self.modrm(0b11, 2, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "jmp {root}")) {
            try self.b(0xe9);
            try self.i32le(@intCast(-(@as(i64, @intCast(self.bytes.items.len)) + 4)));
        } else if (comptime std.mem.eql(u8, instr, "jmp {back}")) {
            try self.b(0xe9);
            const after: i64 = @as(i64, @intCast(self.bytes.items.len)) + 4;
            const target: i64 = @intCast(args[0].handle);
            try self.i32le(@intCast(target - after));
        } else if (comptime std.mem.eql(u8, instr, "ret")) {
            try self.b(0xc3);
        } else if (comptime std.mem.eql(u8, instr, "ud2")) {
            try self.b(0x0f);
            try self.b(0x0b);
        } else if (comptime std.mem.eql(u8, instr, "movq xmm0, {reg}")) {
            try self.b(0x66);
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x6e);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "movq xmm1, {reg}")) {
            try self.b(0x66);
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x6e);
            try self.modrm(0b11, 1, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "movq {reg}, xmm0")) {
            try self.b(0x66);
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x7e);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "addsd xmm0, xmm1")) {
            try self.b(0xf2);
            try self.b(0x0f);
            try self.b(0x58);
            try self.modrm(0b11, 0, 1);
        } else if (comptime std.mem.eql(u8, instr, "subsd xmm0, xmm1")) {
            try self.b(0xf2);
            try self.b(0x0f);
            try self.b(0x5c);
            try self.modrm(0b11, 0, 1);
        } else if (comptime std.mem.eql(u8, instr, "mulsd xmm0, xmm1")) {
            try self.b(0xf2);
            try self.b(0x0f);
            try self.b(0x59);
            try self.modrm(0b11, 0, 1);
        } else if (comptime std.mem.eql(u8, instr, "divsd xmm0, xmm1")) {
            try self.b(0xf2);
            try self.b(0x0f);
            try self.b(0x5e);
            try self.modrm(0b11, 0, 1);
        } else if (comptime std.mem.eql(u8, instr, "ucomisd xmm0, xmm1")) {
            try self.b(0x66);
            try self.b(0x0f);
            try self.b(0x2e);
            try self.modrm(0b11, 0, 1);
        } else if (comptime std.mem.eql(u8, instr, "cmovb {reg}, {reg}")) {
            try self.rex(true, high(args[0]), false, high(args[1]));
            try self.b(0x0f);
            try self.b(0x42);
            try self.modrm(0b11, low(args[0]), low(args[1]));
        } else if (comptime std.mem.eql(u8, instr, "cvtsi2sd xmm0, {reg}")) {
            try self.b(0xf2);
            try self.rex(true, false, false, high(args[0]));
            try self.b(0x0f);
            try self.b(0x2a);
            try self.modrm(0b11, 0, low(args[0]));
        } else if (comptime std.mem.eql(u8, instr, "cvttsd2si {reg}, xmm0")) {
            try self.b(0xf2);
            try self.rex(true, high(args[0]), false, false);
            try self.b(0x0f);
            try self.b(0x2c);
            try self.modrm(0b11, low(args[0]), 0);
        } else {
            @compileError("MachineCode.emit: unknown instruction: " ++ instr);
        }
    }
    // Emits an instruction whose {label} placeholder refers to a not-yet-known
    // forward location, returning a Patch that patch_to_here resolves. All
    // supported forms use a 4-byte rip-relative i32 displacement (the i32 is
    // relative to the byte right after itself).
    fn placeholder(self: *MachineCode, comptime instr: []const u8, args: anytype) !Patch {
        if (comptime std.mem.eql(u8, instr, "jz {label}")) {
            try self.b(0x0f);
            try self.b(0x84);
            const handle: u32 = @intCast(self.bytes.items.len);
            try self.bytes.appendNTimes(self.ally, 0, 4);
            return .{ .handle = handle };
        } else if (comptime std.mem.eql(u8, instr, "jb {label}")) {
            try self.b(0x0f);
            try self.b(0x82);
            const handle: u32 = @intCast(self.bytes.items.len);
            try self.bytes.appendNTimes(self.ally, 0, 4);
            return .{ .handle = handle };
        } else if (comptime std.mem.eql(u8, instr, "jp {label}")) {
            try self.b(0x0f);
            try self.b(0x8a);
            const handle: u32 = @intCast(self.bytes.items.len);
            try self.bytes.appendNTimes(self.ally, 0, 4);
            return .{ .handle = handle };
        } else if (comptime std.mem.eql(u8, instr, "jmp {label}")) {
            try self.b(0xe9);
            const handle: u32 = @intCast(self.bytes.items.len);
            try self.bytes.appendNTimes(self.ally, 0, 4);
            return .{ .handle = handle };
        } else if (comptime std.mem.eql(u8, instr, "lea {reg}, [rip + {label}]")) {
            try self.rex(true, high(args[0]), false, false);
            try self.b(0x8d); // LEA
            try self.modrm(0b00, low(args[0]), 0b101); // RIP-relative
            const handle: u32 = @intCast(self.bytes.items.len);
            try self.bytes.appendNTimes(self.ally, 0, 4);
            return .{ .handle = handle };
        } else {
            @compileError("MachineCode.placeholder: unknown placeholder: " ++ instr);
        }
    }
    fn patch_to_here(self: *MachineCode, p: Patch) !void {
        const target: i64 = @intCast(self.bytes.items.len);
        const after: i64 = @as(i64, @intCast(p.handle)) + 4;
        const delta: i32 = @intCast(target - after);
        std.mem.writeInt(i32, self.bytes.items[p.handle..][0..4], delta, .little);
    }
    fn here(self: *MachineCode) !BackLabel {
        return .{ .handle = @intCast(self.bytes.items.len) };
    }
};

// ===== IR -> Assembly text =================================================
//
// First-cut emitter: walks the cost-annotated IR with a mutable cursor and
// writes Intel-syntax assembly lines into an ArrayList(u8). For each subtree
// emit_expr_to_stack checks whether its reg cost fits in r8..r15; if it does
// it routes to emit_expr_to_reg with a starting register, then pushes the
// resulting register. Otherwise it lowers the node onto the data stack and
// recurses through emit_expr_to_stack for each child.

const Resolver = struct {
    stack: ArrayList([]const u8),
    regs: ArrayList(RegBinding),
    const RegBinding = struct { name: []const u8, reg: Reg };

    fn init() Resolver {
        return .{ .stack = .empty, .regs = .empty };
    }
    fn deinit(self: *Resolver, ally: Ally) void {
        self.stack.deinit(ally);
        self.regs.deinit(ally);
    }

    fn push_anon(self: *Resolver, ally: Ally) !void {
        try self.stack.append(ally, "");
    }
    fn push_named(self: *Resolver, ally: Ally, name: []const u8) !void {
        try self.stack.append(ally, name);
    }
    fn pop_stack(self: *Resolver) void {
        _ = self.stack.pop();
    }
    fn rename_top(self: *Resolver, name: []const u8) void {
        self.stack.items[self.stack.items.len - 1] = name;
    }
    fn bind_reg(self: *Resolver, ally: Ally, name: []const u8, reg: Reg) !void {
        try self.regs.append(ally, .{ .name = name, .reg = reg });
    }
    fn unbind_reg(self: *Resolver) void {
        _ = self.regs.pop();
    }

    const Location = union(enum) { stack: usize, reg: Reg };
    fn lookup(self: *const Resolver, target: []const u8) ?Location {
        var i: usize = self.regs.items.len;
        while (i > 0) {
            i -= 1;
            const e = self.regs.items[i];
            if (std.mem.eql(u8, e.name, target)) return .{ .reg = e.reg };
        }
        i = self.stack.items.len;
        while (i > 0) {
            i -= 1;
            const e = self.stack.items[i];
            if (e.len == 0) continue;
            if (std.mem.eql(u8, e, target)) {
                return .{ .stack = self.stack.items.len - 1 - i };
            }
        }
        return null;
    }
};

fn emit_fun(sink: anytype, ir: Ir, fun: Vm.Fun) !void {
    var resolver = Resolver.init();
    defer resolver.deinit(sink.ally);

    for (fun.args()) |arg| try resolver.push_named(sink.ally, get_symbol(arg));

    var cursor: Index = 0;
    try emit_expr_to_stack(sink, ir.slots, &cursor, &resolver);

    if (fun.args().len > 0) {
        try sink.emit("pop {reg}", .{.rax});
        try sink.emit("add {reg}, {i32}", .{ .rsp, @as(i32, @intCast(fun.args().len * 8)) });
        try sink.emit("push {reg}", .{.rax});
    }
    try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rbp, @as(i32, 0) });
    try sink.emit("add {reg}, {i32}", .{ .rbp, 8 });
    try sink.emit("jmp {reg}", .{.rax});
}
fn emit_expr_to_stack(sink: anytype, slots: []const u64, at: *Index, resolver: *Resolver) error{OutOfMemory}!void {
    const header: Header = @bitCast(slots[at.*]);
    const ally = sink.ally;

    if (header.regs != STACK and header.regs <= 8) {
        const dst: Reg = @enumFromInt(@intFromEnum(Reg.r8));
        try emit_expr_to_reg(sink, slots, at, resolver, dst);
        sink.indent();
        try sink.emit("push {reg}", .{dst});
        sink.deindent();
        try resolver.push_anon(ally);
        return;
    }

    try sink.comment("{s}", .{@tagName(header.tag)});
    sink.indent();
    defer sink.deindent();

    _ = eat_slot(slots, at); // consume header
    switch (header.tag) {
        .word => {
            const word = eat_slot(slots, at);
            const signed = @as(i64, @bitCast(word));
            if (signed >= -0x80 and signed < 0x80) {
                try sink.emit("push {i8}", .{@as(i8, @intCast(signed))});
            } else if (signed >= -0x80000000 and signed < 0x7fffffff) {
                try sink.emit("push {i32}", .{@as(i32, @intCast(signed))});
            } else {
                try sink.emit("mov {reg}, {u64}", .{ .rax, word });
                try sink.emit("push {reg}", .{.rax});
            }
            try resolver.push_anon(ally);
        },
        .object => {
            const a = eat_slot(slots, at);
            try sink.emit("mov {reg}, {u64}", .{ .rax, a });
            try sink.emit("push {reg}", .{.rax});
            try resolver.push_anon(ally);
        },
        .name => {
            const name = get_symbol(Obj{ .address = eat_slot(slots, at) });
            const loc = resolver.lookup(name) orelse unreachable;
            switch (loc) {
                .stack => |o| try sink.emit("push [{reg} + {i32}]", .{ .rsp, @as(i32, @intCast(o * 8)) }),
                .reg => |r| try sink.emit("push {reg}", .{r}),
            }
            try resolver.push_anon(ally);
        },
        .let => {
            const name = get_symbol(Obj{ .address = eat_slot(slots, at) });
            try emit_expr_to_stack(sink, slots, at, resolver); // def
            resolver.rename_top(name);
            try emit_expr_to_stack(sink, slots, at, resolver); // body
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("add {reg}, {i32}", .{ .rsp, 8 });
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .also => {
            try emit_expr_to_stack(sink, slots, at, resolver); // ignored
            try sink.emit("add {reg}, {i32}", .{ .rsp, 8 });
            resolver.pop_stack();
            try emit_expr_to_stack(sink, slots, at, resolver); // value
        },
        .add, .subtract, .and_, .or_, .xor => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            switch (header.tag) {
                .add => try sink.emit("add {reg}, {reg}", .{ .rax, .rbx }),
                .subtract => try sink.emit("sub {reg}, {reg}", .{ .rax, .rbx }),
                .and_ => try sink.emit("and {reg}, {reg}", .{ .rax, .rbx }),
                .or_ => try sink.emit("or {reg}, {reg}", .{ .rax, .rbx }),
                .xor => try sink.emit("xor {reg}, {reg}", .{ .rax, .rbx }),
                else => unreachable,
            }
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .multiply => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("imul {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .divide, .modulo => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            // cqo/idiv clobber rdx (cqo sign-extends rax into rdx, idiv leaves
            // the remainder there), but rdx is reserved for the original rsp
            // marker in our calling convention. Stash it in r8 across the
            // divide and restore after we've captured the result.
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdx });
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("cqo", .{});
            try sink.emit("idiv {reg}", .{.rbx});
            if (header.tag == .modulo) {
                // idiv leaves the *truncated* remainder in rdx; convert it to
                // the *floor* remainder (Zig @mod semantics) by adding the
                // divisor when sign(rdx) != sign(rbx) and rdx != 0.
                try sink.emit("mov {reg}, {reg}", .{ .rax, .rdx });
                try sink.emit("xor {reg}, {reg}", .{ .rax, .rbx });
                try sink.emit("sar {reg}, {u8}", .{ .rax, @as(u8, 63) }); // -1 if signs differ, else 0
                try sink.emit("and {reg}, {reg}", .{ .rax, .rbx }); // = divisor or 0
                try sink.emit("mov {reg}, {u64}", .{ .r9, @as(u64, 0) });
                try sink.emit("test {reg}, {reg}", .{ .rdx, .rdx });
                try sink.emit("cmovz {reg}, {reg}", .{ .rax, .r9 }); // if rdx==0, no adjust
                try sink.emit("add {reg}, {reg}", .{ .rdx, .rax });
            }
            try sink.emit("push {reg}", .{if (header.tag == .divide) Reg.rax else Reg.rdx});
            try sink.emit("mov {reg}, {reg}", .{ .rdx, .r8 });
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .shift_left, .shift_right => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            // x86's variable-count shifts require the count in cl, but rcx
            // holds the sandbox cursor. So:
            // 1. stash rcx in rbx
            // 2. do the shift
            // 3. restore
            try sink.emit("mov {reg}, {reg}", .{ .rbx, .rcx });
            try sink.emit("pop {reg}", .{.rcx});
            try sink.emit("pop {reg}", .{.rax});
            if (header.tag == .shift_left)
                try sink.emit("shl {reg}, cl", .{.rax})
            else
                try sink.emit("sar {reg}, cl", .{.rax});
            try sink.emit("mov {reg}, {reg}", .{ .rcx, .rbx });
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .compare => { // 1=equal, 2=greater, 4=less
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("cmp {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 1) });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 2) });
            try sink.emit("cmovg {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 4) });
            try sink.emit("cmovl {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_add => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("movq xmm1, {reg}", .{.rbx});
            try sink.emit("addsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_subtract => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("movq xmm1, {reg}", .{.rbx});
            try sink.emit("subsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_multiply => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("movq xmm1, {reg}", .{.rbx});
            try sink.emit("mulsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_divide => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("movq xmm1, {reg}", .{.rbx});
            try sink.emit("divsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_compare => {
            // Result: 1 if bit-equal OR unordered (NaN involved), 2 if a > b,
            // 4 if a < b, 1 if numerically equal but not bit-equal (e.g. +0/-0).
            // Bit-equal fast path handles NaN==NaN (same bits) returning 1.
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx}); // b
            try sink.emit("pop {reg}", .{.rax}); // a
            try sink.emit("mov {reg}, {u64}", .{ .r8, @as(u64, 1) });
            try sink.emit("cmp {reg}, {reg}", .{ .rax, .rbx });
            const equal_jmp = try sink.placeholder("jz {label}", .{});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("movq xmm1, {reg}", .{.rbx});
            try sink.emit("ucomisd xmm0, xmm1", .{});
            const unord_jmp = try sink.placeholder("jp {label}", .{});
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 2) });
            try sink.emit("cmova {reg}, {reg}", .{ .r8, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 4) });
            try sink.emit("cmovb {reg}, {reg}", .{ .r8, .rbx });
            try sink.patch_to_here(equal_jmp);
            try sink.patch_to_here(unord_jmp);
            try sink.emit("push {reg}", .{.r8});
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .f_is_finite => {
            // Finite iff masked-off-sign bits < 0x7FF0_0000_0000_0000
            // (which is +inf; equal-or-greater is inf or NaN).
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 0x7fffffffffffffff) });
            try sink.emit("and {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 0x7ff0000000000000) });
            try sink.emit("cmp {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 0) });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("cmovb {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("push {reg}", .{.rax});
        },
        .int_to_float => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("cvtsi2sd xmm0, {reg}", .{.rax});
            try sink.emit("movq {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
        },
        .float_to_int => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("movq xmm0, {reg}", .{.rax});
            try sink.emit("cvttsd2si {reg}, xmm0", .{.rax});
            try sink.emit("push {reg}", .{.rax});
        },
        .if_ => {
            try emit_expr_to_stack(sink, slots, at, resolver); // cond
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("test {reg}, {reg}", .{ .rax, .rax });
            const jz = try sink.placeholder("jz {label}", .{});
            resolver.pop_stack();
            try emit_expr_to_stack(sink, slots, at, resolver); // then
            const jmp = try sink.placeholder("jmp {label}", .{});
            try sink.patch_to_here(jz);
            resolver.pop_stack();
            try emit_expr_to_stack(sink, slots, at, resolver); // else
            try sink.patch_to_here(jmp);
        },
        .points => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rax, 0 });
            try sink.emit("shr {reg}, {u8}", .{ .rax, @as(u8, 63) });
            try sink.emit("push {reg}", .{.rax});
        },
        .size => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rax, 0 });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, 0xffffffffffff });
            try sink.emit("and {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("push {reg}", .{.rax});
        },
        .load => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rbx});
            try sink.emit("shl {reg}, {u8}", .{ .rbx, @as(u8, 3) });
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("add {reg}, {i32}", .{ .rax, 8 });
            try sink.emit("add {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("push [{reg} + {i32}]", .{ .rax, 0 });
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .load_imm => {
            const imm = eat_slot(slots, at);
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("push [{reg} + {i32}]", .{ .rax, @as(i32, @intCast(imm * 8 + 8)) });
        },
        .new_leaf, .new_inner => {
            const n: usize = @intCast(header.rest);
            for (0..n) |_| try emit_expr_to_stack(sink, slots, at, resolver);
            const header_val: u64 = if (header.tag == .new_inner)
                (@as(u64, 1) << 63) | @as(u64, n)
            else
                @as(u64, n);
            try sink.emit("mov {reg}, {reg}", .{ .rax, .rdi });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, header_val });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx });
            for (0..n) |i| {
                const src_off: i32 = @intCast((n - 1 - i) * 8);
                const dst_off: i32 = @intCast((i + 1) * 8);
                try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rsp, src_off });
                try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, dst_off, .rbx });
            }
            try sink.emit("add {reg}, {i32}", .{ .rdi, @as(i32, @intCast((n + 1) * 8)) });
            if (n > 0) try sink.emit("add {reg}, {i32}", .{ .rsp, @as(i32, @intCast(n * 8)) });
            try sink.emit("push {reg}", .{.rax});
            for (0..n) |_| resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .flatten_to_leaf => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdi });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 8 });
            try sink.emit("xor {reg}, {reg}", .{ .r9, .r9 });
            const loop = try sink.here();
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rax, 0 });
            try sink.emit("mov {reg}, {u64}", .{ .r10, @as(u64, 0xffffffffffff) });
            try sink.emit("and {reg}, {reg}", .{ .rbx, .r10 });
            const done = try sink.placeholder("jz {label}", .{});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .r10, .rax, 8 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .r11, .r10, 8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .r11 });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 8 });
            try sink.emit("add {reg}, {i32}", .{ .r9, 1 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rax, 16 });
            try sink.emit("jmp {back}", .{loop});
            try sink.patch_to_here(done);
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .r8, 0, .r9 });
            try sink.emit("push {reg}", .{.r8});
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .flatten_to_inner => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdi });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 8 });
            try sink.emit("xor {reg}, {reg}", .{ .r9, .r9 });
            const loop = try sink.here();
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rax, 0 });
            try sink.emit("mov {reg}, {u64}", .{ .r10, @as(u64, 0xffffffffffff) });
            try sink.emit("and {reg}, {reg}", .{ .rbx, .r10 });
            const done = try sink.placeholder("jz {label}", .{});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .r10, .rax, 8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .r10 });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 8 });
            try sink.emit("add {reg}, {i32}", .{ .r9, 1 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rax, 16 });
            try sink.emit("jmp {back}", .{loop});
            try sink.patch_to_here(done);
            try sink.emit("mov {reg}, {u64}", .{ .r10, @as(u64, 1) << 63 });
            try sink.emit("or {reg}, {reg}", .{ .r9, .r10 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .r8, 0, .r9 });
            try sink.emit("push {reg}", .{.r8});
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .call => {
            const n: usize = @intCast(header.rest);
            for (0..n) |_| try emit_expr_to_stack(sink, slots, at, resolver);
            try emit_expr_to_stack(sink, slots, at, resolver); // callee obj
            try sink.emit("pop {reg}", .{.rax});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 0, .rdi });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 8, .rsp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 16, .rbp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 24, .rcx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 32, .rsi });
            try sink.emit("mov {reg}, {reg}", .{ .rsp, .rdx });
            try sink.emit("push {reg}", .{.rdx}); // save rdx; also keeps rsp 16-aligned for the call
            try sink.emit("mov {reg}, {reg}", .{ .rdi, .rbx }); // arg 0
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rdx, 0 }); // arg 1
            try sink.emit("mov {reg}, {reg}", .{ .rdx, .rax }); // arg 2: callee_obj
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, @intFromPtr(&hook_compile)) });
            try sink.emit("call {reg}", .{.rax});
            // rax now holds the machine-code pointer (SysV return). Restore.
            try sink.emit("pop {reg}", .{.rdx});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rdi, .rbx, 0 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rbx, 16 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rcx, .rbx, 24 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rbx, 32 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rbx, 8 });
            try sink.emit("sub {reg}, {i32}", .{ .rbp, 8 });
            const ret_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbp, 0, .rbx });
            try sink.emit("jmp {reg}", .{.rax});
            try sink.patch_to_here(ret_patch);
            for (0..n) |_| resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .call_imm => {
            const callee = eat_slot(slots, at);
            const n: usize = @intCast(header.rest);
            for (0..n) |_| try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("sub {reg}, {i32}", .{ .rbp, 8 });
            const ret_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbp, 0, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rax, callee });
            try sink.emit("jmp {reg}", .{.rax});
            try sink.patch_to_here(ret_patch);
            for (0..n) |_| resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .call_indirect => {
            try emit_expr_to_stack(sink, slots, at, resolver); // args
            try emit_expr_to_stack(sink, slots, at, resolver); // fun
            // Stack: [..., args_obj, fun_obj], top = fun_obj.
            try sink.emit("pop {reg}", .{.r12}); // r12 = fun_obj (preserved across the loop)
            try sink.emit("pop {reg}", .{.rbx}); // rbx = args_obj
            // n = args_obj.size() = header & 0xffffffffffff.
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .r10, .rbx, 0 });
            try sink.emit("mov {reg}, {u64}", .{ .r11, @as(u64, 0xffffffffffff) });
            try sink.emit("and {reg}, {reg}", .{ .r10, .r11 });
            try sink.emit("add {reg}, {i32}", .{ .rbx, 8 }); // rbx -> first child slot
            // Push each child onto the data stack in order, so child[n-1] is on top.
            const loop = try sink.here();
            try sink.emit("test {reg}, {reg}", .{ .r10, .r10 });
            const done = try sink.placeholder("jz {label}", .{});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .r11, .rbx, 0 });
            try sink.emit("push {reg}", .{.r11});
            try sink.emit("add {reg}, {i32}", .{ .rbx, 8 });
            try sink.emit("sub {reg}, {i32}", .{ .r10, 1 });
            try sink.emit("jmp {back}", .{loop});
            try sink.patch_to_here(done);
            // Bridge to SysV and call hook_compile(*VmState, *ZigState, fun_obj).
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 0, .rdi });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 8, .rsp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 16, .rbp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 24, .rcx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 32, .rsi });
            try sink.emit("mov {reg}, {reg}", .{ .rsp, .rdx });
            try sink.emit("push {reg}", .{.rdx}); // save rdx; also keeps rsp 16-aligned for the call
            try sink.emit("mov {reg}, {reg}", .{ .rdi, .rbx }); // arg 0: *VmState
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rdx, 0 }); // arg 1: *ZigState
            try sink.emit("mov {reg}, {reg}", .{ .rdx, .r12 }); // arg 2: fun_obj
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, @intFromPtr(&hook_compile)) });
            try sink.emit("call {reg}", .{.rax});
            // rax now holds the callee's machine-code pointer.
            try sink.emit("pop {reg}", .{.rdx});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rdi, .rbx, 0 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rbx, 16 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rcx, .rbx, 24 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rbx, 32 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rbx, 8 });
            // Push return address onto the call stack and jump to the callee.
            try sink.emit("sub {reg}, {i32}", .{ .rbp, 8 });
            const ret_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbp, 0, .rbx });
            try sink.emit("jmp {reg}", .{.rax});
            try sink.patch_to_here(ret_patch);
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .rec => {
            const n: usize = @intCast(header.rest);
            for (0..n) |_| try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("sub {reg}, {i32}", .{ .rbp, 8 });
            const ret_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbp, 0, .rbx });
            try sink.emit("jmp {root}", .{});
            try sink.patch_to_here(ret_patch);
            for (0..n) |_| resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .collect_garbage => {
            try sink.emit("push {reg}", .{.rdi}); // heap checkpoint
            try resolver.push_anon(ally);
            try emit_expr_to_stack(sink, slots, at, resolver); // keep
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 0, .rdi });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 8, .rsp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 16, .rbp });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 24, .rcx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rbx, 32, .rsi });
            try sink.emit("mov {reg}, {reg}", .{ .rsp, .rdx });
            try sink.emit("push {reg}", .{.rdx});
            try sink.emit("mov {reg}, {reg}", .{ .rdi, .rbx }); // arg 0: *VmState
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rdx, 0 }); // arg 1: *ZigState
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, @intFromPtr(&hook_gc)) });
            try sink.emit("call {reg}", .{.rax});
            // (the garbage collection runs)
            try sink.emit("pop {reg}", .{.rdx});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rdx, 8 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rdi, .rbx, 0 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rbx, 16 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rcx, .rbx, 24 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rbx, 32 });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rbx, 8 });
            resolver.pop_stack();
            resolver.pop_stack();
            try resolver.push_anon(ally);
        },
        .sandbox => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("pop {reg}", .{.rax}); // rax = limit
            // Cap fuel: rsi = min(rsi, limit). r8 captures the diff so we can
            // add it back on sandbox exit (normal or crash). Use cmova (above,
            // unsigned) — cmovg treats Vm.max_fuel (= u64 max = -1 signed) as
            // smaller than any limit and silently skips the cap.
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rsi }); // r8 = original fuel
            try sink.emit("cmp {reg}, {reg}", .{ .rsi, .rax });
            try sink.emit("cmova {reg}, {reg}", .{ .rsi, .rax }); // if rsi>limit (unsigned): rsi=limit
            try sink.emit("sub {reg}, {reg}", .{ .r8, .rsi }); // r8 = original - capped
            resolver.pop_stack();

            // Push a sandbox entry: { crash_target, heap, call, data, fuel_diff, ouf_target }.
            try sink.emit("sub {reg}, {i32}", .{ .rcx, 48 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 32, .r8 }); // fuel_diff
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 24, .rsp }); // data snapshot
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 16, .rbp }); // call snapshot
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 8, .rdi }); // heap snapshot
            const crash_target_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 0, .rbx }); // crash_target
            const ouf_target_patch = try sink.placeholder("lea {reg}, [rip + {label}]", .{.rbx});
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rcx, 40, .rbx }); // ouf_target

            // Body runs; on success, leaves its result on top of the data stack.
            try emit_expr_to_stack(sink, slots, at, resolver);

            // Normal exit: pop entry, restore fuel, wrap as (leaf 0, result).
            try sink.emit("pop {reg}", .{.rax}); // rax = body result
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 32 });
            try sink.emit("add {reg}, {reg}", .{ .rsi, .rbx }); // restore fuel
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop entry
            // Allocate tag leaf (header=1, word=0).
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdi }); // r8 = &tag_leaf
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx }); // header
            try sink.emit("xor {reg}, {reg}", .{ .rbx, .rbx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .rbx }); // word = 0
            try sink.emit("add {reg}, {i32}", .{ .rdi, 16 });
            // Allocate wrapper inner(tag_leaf, result).
            try sink.emit("mov {reg}, {reg}", .{ .r9, .rdi }); // r9 = &wrapper
            try sink.emit("mov {reg}, {u64}", .{ .rbx, (@as(u64, 1) << 63) | 2 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx }); // header
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .r8 }); // child 0
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 16, .rax }); // child 1
            try sink.emit("add {reg}, {i32}", .{ .rdi, 24 });
            try sink.emit("push {reg}", .{.r9});
            const join_patch_normal = try sink.placeholder("jmp {label}", .{});

            // Crash exit: .crash has already restored data/call/fuel and popped
            // our entry, and pushed the error on top of the restored data stack.
            try sink.patch_to_here(crash_target_patch);
            try sink.emit("pop {reg}", .{.rax}); // rax = error
            // Allocate tag leaf (header=1, word=1).
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdi });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx }); // header
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .rbx }); // word = 1
            try sink.emit("add {reg}, {i32}", .{ .rdi, 16 });
            // Allocate wrapper inner(tag_leaf, error).
            try sink.emit("mov {reg}, {reg}", .{ .r9, .rdi });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, (@as(u64, 1) << 63) | 2 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .r8 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 16, .rax });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 24 });
            try sink.emit("push {reg}", .{.r9});
            const join_patch_crash = try sink.placeholder("jmp {label}", .{});

            // Out-of-fuel exit: .use_fuel's unwinder has already restored
            // data/call/fuel and popped our entry. No value on top of the
            // data stack (use_fuel doesn't push one). Wrap as (leaf 2,) —
            // a single-child inner, matching what the byte-code emits.
            try sink.patch_to_here(ouf_target_patch);
            // Allocate tag leaf (header=1, word=2).
            try sink.emit("mov {reg}, {reg}", .{ .r8, .rdi });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx }); // header
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 2) });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .rbx }); // word = 2
            try sink.emit("add {reg}, {i32}", .{ .rdi, 16 });
            // Allocate wrapper inner(tag_leaf) — single child.
            try sink.emit("mov {reg}, {reg}", .{ .r9, .rdi });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, (@as(u64, 1) << 63) | 1 });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 8, .r8 });
            try sink.emit("add {reg}, {i32}", .{ .rdi, 16 });
            try sink.emit("push {reg}", .{.r9});

            try sink.patch_to_here(join_patch_normal);
            try sink.patch_to_here(join_patch_crash);

            resolver.pop_stack(); // body result consumed by the wrap
            try resolver.push_anon(ally); // sandbox result pushed
        },
        .use_fuel => {
            // Charge `amount` against rsi. The sub may underflow; on borrow
            // (i.e. rsi < amount before the sub) we unwind to the nearest
            // sandbox with a non-zero fuel_diff.
            try emit_expr_to_stack(sink, slots, at, resolver); // amount
            try sink.emit("pop {reg}", .{.rax});
            resolver.pop_stack();
            try sink.emit("sub {reg}, {reg}", .{ .rsi, .rax });
            const ouf_unwind_patch = try sink.placeholder("jb {label}", .{});
            try emit_expr_to_stack(sink, slots, at, resolver); // body (continues normally)
            const past_unwind_patch = try sink.placeholder("jmp {label}", .{});

            // Out-of-fuel unwind.
            try sink.patch_to_here(ouf_unwind_patch);
            const loop = try sink.here();
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 32 });
            try sink.emit("test {reg}, {reg}", .{ .rbx, .rbx });
            const keep_popping_patch = try sink.placeholder("jz {label}", .{});
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rcx, 24 }); // data
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rcx, 16 }); // call
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsi, .rcx, 32 }); // rsi = fuel_diff
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 40 }); // on_out_of_fuel
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop entry
            try sink.emit("jmp {reg}", .{.rbx});
            try sink.patch_to_here(keep_popping_patch);
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop this entry, keep walking
            try sink.emit("jmp {back}", .{loop});

            try sink.patch_to_here(past_unwind_patch);
        },
        .crash => {
            try emit_expr_to_stack(sink, slots, at, resolver);
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rsp, 0 }); // rax = error
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rcx, 24 }); // rsp = sandbox.data
            try sink.emit("push {reg}", .{.rax}); // error onto restored stack
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rcx, 16 }); // rbp = sandbox.call
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 32 }); // rbx = fuel_diff
            try sink.emit("add {reg}, {reg}", .{ .rsi, .rbx }); // restore fuel
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 0 }); // rbx = crash_target
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop entry
            try sink.emit("jmp {reg}", .{.rbx});
            // We deliberately leave the err's anon on the resolver instead of
            // pop+push'ing a "result" anon. The surrounding stack-path parent
            // (`.if_`, `.add`, `.let`, etc.) is going to consume exactly one
            // anon as our "result" — the err anon stands in for that on the
            // non-crashing branch and the machine code never reaches the
            // parent's post-child code on the crashing branch, so the resolver
            // ends up at the correct depth for whatever code is emitted next.
        },
        .unreachable_ => {
            try sink.emit("ud2", .{});
            try resolver.push_anon(ally);
        },
    }
}
fn emit_expr_to_reg(sink: anytype, slots: []const u64, at: *Index, resolver: *Resolver, dst: Reg) error{OutOfMemory}!void {
    const header: Header = @bitCast(eat_slot(slots, at));

    try sink.comment("{s}", .{@tagName(header.tag)});
    sink.indent();
    defer sink.deindent();

    switch (header.tag) {
        .word => {
            const w = eat_slot(slots, at);
            try sink.emit("mov {reg}, {u64}", .{ dst, w });
        },
        .object => {
            const a = eat_slot(slots, at);
            try sink.emit("mov {reg}, {u64}", .{ dst, a });
        },
        .name => {
            const name = get_symbol(Obj{ .address = eat_slot(slots, at) });
            const loc = resolver.lookup(name) orelse unreachable;
            switch (loc) {
                .stack => |o| try sink.emit("mov {reg}, [{reg} + {i32}]", .{ dst, .rsp, @as(i32, @intCast(o * 8)) }),
                .reg => |r| try sink.emit("mov {reg}, {reg}", .{ dst, r }),
            }
        },
        .let => {
            const name = get_symbol(Obj{ .address = eat_slot(slots, at) });
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try resolver.bind_reg(sink.ally, name, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            resolver.unbind_reg();
            try sink.emit("mov {reg}, {reg}", .{ dst, dst.next() });
        },
        .also => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
        },
        .add, .subtract, .and_, .or_, .xor => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            switch (header.tag) {
                .add => try sink.emit("add {reg}, {reg}", .{ dst, dst.next() }),
                .subtract => try sink.emit("sub {reg}, {reg}", .{ dst, dst.next() }),
                .and_ => try sink.emit("and {reg}, {reg}", .{ dst, dst.next() }),
                .or_ => try sink.emit("or {reg}, {reg}", .{ dst, dst.next() }),
                .xor => try sink.emit("xor {reg}, {reg}", .{ dst, dst.next() }),
                else => unreachable,
            }
        },
        .multiply => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("imul {reg}, {reg}", .{ dst, dst.next() });
        },
        .divide, .modulo => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            // cqo/idiv clobber rdx; save it in rbx (intra-instruction temp)
            // and restore after capturing the result. See the stack-path
            // variant for the full explanation, including the floor-mod
            // adjustment for .modulo.
            try sink.emit("mov {reg}, {reg}", .{ .rbx, .rdx });
            try sink.emit("mov {reg}, {reg}", .{ .rax, dst });
            try sink.emit("cqo", .{});
            try sink.emit("idiv {reg}", .{dst.next()});
            if (header.tag == .modulo) {
                try sink.emit("mov {reg}, {reg}", .{ .rax, .rdx });
                try sink.emit("xor {reg}, {reg}", .{ .rax, dst.next() });
                try sink.emit("sar {reg}, {u8}", .{ .rax, @as(u8, 63) });
                try sink.emit("and {reg}, {reg}", .{ .rax, dst.next() });
                try sink.emit("mov {reg}, {u64}", .{ dst, @as(u64, 0) });
                try sink.emit("test {reg}, {reg}", .{ .rdx, .rdx });
                try sink.emit("cmovz {reg}, {reg}", .{ .rax, dst });
                try sink.emit("add {reg}, {reg}", .{ .rdx, .rax });
            }
            try sink.emit("mov {reg}, {reg}", .{ dst, if (header.tag == .divide) Reg.rax else Reg.rdx });
            try sink.emit("mov {reg}, {reg}", .{ .rdx, .rbx });
        },
        .shift_left, .shift_right => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            // Stash rcx (the sandbox cursor) in rbx while we use cl for the
            // shift count; restore it before the surrounding sees rcx.
            try sink.emit("mov {reg}, {reg}", .{ .rbx, .rcx });
            try sink.emit("mov {reg}, {reg}", .{ .rcx, dst.next() });
            if (header.tag == .shift_left)
                try sink.emit("shl {reg}, cl", .{dst})
            else
                try sink.emit("sar {reg}, cl", .{dst});
            try sink.emit("mov {reg}, {reg}", .{ .rcx, .rbx });
        },
        .compare => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("cmp {reg}, {reg}", .{ dst, dst.next() });
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 1) });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 2) });
            try sink.emit("cmovg {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 4) });
            try sink.emit("cmovl {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {reg}", .{ dst, .rax });
        },
        .f_add => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("movq xmm1, {reg}", .{dst.next()});
            try sink.emit("addsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{dst});
        },
        .f_subtract => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("movq xmm1, {reg}", .{dst.next()});
            try sink.emit("subsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{dst});
        },
        .f_multiply => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("movq xmm1, {reg}", .{dst.next()});
            try sink.emit("mulsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{dst});
        },
        .f_divide => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("movq xmm1, {reg}", .{dst.next()});
            try sink.emit("divsd xmm0, xmm1", .{});
            try sink.emit("movq {reg}, xmm0", .{dst});
        },
        .f_compare => {
            // See the stack-variant for the algorithm. Result accumulates in
            // rax (intra-instruction temp); dst gets the final value.
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("mov {reg}, {u64}", .{ .rax, @as(u64, 1) });
            try sink.emit("cmp {reg}, {reg}", .{ dst, dst.next() });
            const equal_jmp = try sink.placeholder("jz {label}", .{});
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("movq xmm1, {reg}", .{dst.next()});
            try sink.emit("ucomisd xmm0, xmm1", .{});
            const unord_jmp = try sink.placeholder("jp {label}", .{});
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 2) });
            try sink.emit("cmova {reg}, {reg}", .{ .rax, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 4) });
            try sink.emit("cmovb {reg}, {reg}", .{ .rax, .rbx });
            try sink.patch_to_here(equal_jmp);
            try sink.patch_to_here(unord_jmp);
            try sink.emit("mov {reg}, {reg}", .{ dst, .rax });
        },
        .f_is_finite => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 0x7fffffffffffffff) });
            try sink.emit("and {reg}, {reg}", .{ dst, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 0x7ff0000000000000) });
            try sink.emit("cmp {reg}, {reg}", .{ dst, .rbx });
            try sink.emit("mov {reg}, {u64}", .{ dst, @as(u64, 0) });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, @as(u64, 1) });
            try sink.emit("cmovb {reg}, {reg}", .{ dst, .rbx });
        },
        .int_to_float => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("cvtsi2sd xmm0, {reg}", .{dst});
            try sink.emit("movq {reg}, xmm0", .{dst});
        },
        .float_to_int => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("movq xmm0, {reg}", .{dst});
            try sink.emit("cvttsd2si {reg}, xmm0", .{dst});
        },
        .if_ => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("test {reg}, {reg}", .{ dst, dst });
            const jz = try sink.placeholder("jz {label}", .{});
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            const jmp = try sink.placeholder("jmp {label}", .{});
            try sink.patch_to_here(jz);
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.patch_to_here(jmp);
        },
        .new_leaf, .new_inner => {
            const n: usize = @intCast(header.rest);
            for (0..n) |i| {
                try emit_expr_to_reg(sink, slots, at, resolver, dst.add(i));
            }
            for (0..n) |i| {
                const off: i32 = @intCast((i + 1) * 8);
                try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, off, dst.add(i) });
            }
            const header_val: u64 = if (header.tag == .new_inner)
                (@as(u64, 1) << 63) | @as(u64, n)
            else
                @as(u64, n);
            try sink.emit("mov {reg}, {u64}", .{ .rbx, header_val });
            try sink.emit("mov [{reg} + {i32}], {reg}", .{ .rdi, 0, .rbx });
            try sink.emit("mov {reg}, {reg}", .{ dst, .rdi });
            try sink.emit("add {reg}, {i32}", .{ .rdi, @as(i32, @intCast((n + 1) * 8)) });
        },
        .points => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ dst, dst, 0 });
            try sink.emit("shr {reg}, {u8}", .{ dst, @as(u8, 63) });
        },
        .size => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ dst, dst, 0 });
            try sink.emit("mov {reg}, {u64}", .{ .rbx, 0xffffffffffff });
            try sink.emit("and {reg}, {reg}", .{ dst, .rbx });
        },
        .load => {
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try emit_expr_to_reg(sink, slots, at, resolver, dst.next());
            try sink.emit("shl {reg}, {u8}", .{ dst.next(), @as(u8, 3) });
            try sink.emit("add {reg}, {i32}", .{ dst, 8 });
            try sink.emit("add {reg}, {reg}", .{ dst, dst.next() });
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ dst, dst, 0 });
        },
        .load_imm => {
            const imm = eat_slot(slots, at);
            try emit_expr_to_reg(sink, slots, at, resolver, dst);
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ dst, dst, @as(i32, @intCast(imm * 8 + 8)) });
        },
        .call, .call_imm, .call_indirect, .rec, .collect_garbage, .sandbox, .use_fuel => unreachable,
        .crash => {
            // Reg-path callers expect their children to be resolver-neutral
            // (the result lives in `dst`, not on the data stack; the
            // `emit_expr_to_stack` wrapper that delegates here does its own
            // `push_anon` afterwards). emit_expr_to_stack(err) above pushed an
            // anon onto the resolver, so we pop it back off to land neutral.
            // The stack-path .crash variant leaves the err anon in place
            // instead, because its parent's bookkeeping pops one anon as the
            // "child's result" — different convention, same end depth.
            try emit_expr_to_stack(sink, slots, at, resolver);
            resolver.pop_stack();
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rax, .rsp, 0 }); // error
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rsp, .rcx, 24 }); // restore data stack
            try sink.emit("push {reg}", .{.rax}); // push error
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbp, .rcx, 16 }); // restore call stack
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 32 }); // fuel_diff
            try sink.emit("add {reg}, {reg}", .{ .rsi, .rbx }); // restore fuel
            try sink.emit("mov {reg}, [{reg} + {i32}]", .{ .rbx, .rcx, 0 }); // on crash handler
            try sink.emit("add {reg}, {i32}", .{ .rcx, 48 }); // pop sandbox stack entry
            try sink.emit("jmp {reg}", .{.rbx}); // jump to crash handler
        },
        .unreachable_ => try sink.emit("ud2", .{}),
        else => @panic(@tagName(header.tag)),
    }
}

// Garbage collection and deduplication remove the jit-compiled machine code of
// moved functions (they may contain hardcoded addresses of moved objects). They
// will just be jit-compiled the next time they are called.
// Over time, long-used functions (and other long-lived objects) will be at the
// beginning of the heap and no longer affected by garbage collection,
// preventing unnecessary recompiles.

pub fn garbage_collect(vm: *Self, checkpoint: Heap.Checkpoint, keep: Obj) !Obj {
    const result = try vm.heap.garbage_collect(vm.ally, checkpoint, keep);
    vm.jit_cache.remove_everything_after(result.stable_until);
    return result.keep;
}

pub fn deduplicate(vm: *Self, checkpoint: Heap.Checkpoint, obj: Obj) !Obj {
    var map = try vm.heap.deduplicate(vm.ally, checkpoint);
    defer map.deinit();
    vm.jit_cache.remove_everything_after(checkpoint.address);
    return map.get(obj) orelse obj;
}

// Next up: Hooks. Zig functions that follow the SysV calling convention so I
// can call them from assembly.

fn hook_gc(vm_state: *VmState, zig_state: *ZigState) callconv(.c) void {
    const vm = zig_state.vm;
    vm_state.store_to_vm(vm);
    defer vm_state.load_from_vm(vm);

    const keep = Obj{ .address = vm.data_stack.pop() };
    const checkpoint = Heap.Checkpoint{ .address = vm.data_stack.pop() };
    const mapped = vm.garbage_collect(checkpoint, keep) catch @panic("gc OOM");
    vm.data_stack.push(mapped.address) catch @panic("gc stack overflow");
}

// Looks up (or compiles) the callee's machine code and returns a pointer to
// the first byte.
fn hook_compile(vm_state: *VmState, zig_state: *ZigState, callee_addr: u64) callconv(.c) [*]const u8 {
    const vm = zig_state.vm;
    vm_state.store_to_vm(vm);
    defer vm_state.load_from_vm(vm);

    const callee = Vm.Fun{ .obj = .{ .address = callee_addr } };
    const machine_code = vm.compile(callee) catch @panic("hook_compile: compile failed");
    return machine_code.ptr;
}
