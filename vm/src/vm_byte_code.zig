const std = @import("std");
const ArrayList = std.ArrayList;
const Map = std.AutoArrayHashMap;
const Ally = std.mem.Allocator;

const Heap = @import("heap.zig");
const Word = Heap.Word;
const Obj = Heap.Obj;
const ObjMap = @import("obj_map.zig").ObjMap;
const Vm = @import("vm.zig");
const get_symbol = Heap.get_symbol;

const Self = @This();

ally: Ally,
heap: *Heap,
data_stack: Stack,
call_stack: std.ArrayList(Ip) = .empty,
compiled_cache: ObjMap(CompiledFun),
instruction_count: usize = 0,
fuel: usize = 0,
sandbox_stack: std.ArrayList(SandboxScope) = .empty,
crash_payload: ?Obj = null,

// Ip = instruction pointer
const Ip = struct { instructions: []const Instruction, index: usize };

const SandboxScope = struct {
    data_stack_size: usize,
    call_stack_size: usize,
    heap_checkpoint: Heap.Checkpoint,
    fuel_diff: usize,
    on_crash: Ip,
    on_out_of_fuel: Ip,
};

const Stack = struct {
    memory: []Word,
    used: usize,

    pub fn init(ally: Ally, size: usize) !Stack {
        return .{ .memory = try ally.alloc(Word, size), .used = 0 };
    }
    pub fn push(self: *Stack, word: Word) !void {
        if (self.memory.len == self.used) @panic("stack overflow");
        self.memory[self.used] = word;
        self.used += 1;
    }
    pub fn push_int(self: *Stack, int: i64) !void {
        try self.push(@bitCast(int));
    }
    pub fn push_float(self: *Stack, float: f64) !void {
        try self.push(@bitCast(float));
    }
    pub fn push_obj(self: *Stack, obj: Obj) !void {
        try self.push(obj.address);
    }
    pub fn pop(self: *Stack) Word {
        self.used -= 1;
        return self.memory[self.used];
    }
    pub fn pop_int(self: *Stack) i64 {
        return @bitCast(self.pop());
    }
    pub fn pop_float(self: *Stack) f64 {
        return @bitCast(self.pop());
    }
    pub fn pop_obj(self: *Stack) Obj {
        return .{ .address = self.pop() };
    }
    pub fn pop_n(self: *Stack, n: usize) void {
        self.used -= n;
    }
    pub fn get(self: *Stack, offset: usize) Word {
        return self.memory[self.used - 1 - offset];
    }
    pub fn is_empty(self: Stack) bool {
        return self.used == 0;
    }
};

pub const CompiledFun = struct {
    instructions: []const Instruction,

    pub fn format(fun: CompiledFun, writer: *std.io.Writer) !void {
        try writer.print("compiled fun", .{});
        for (fun.instructions, 0..) |instr, i| {
            try writer.print("\n{d:4} {f}", .{ i, instr });
        }
    }
};
pub const Instruction = union(enum) {
    word: Word,
    // Pushes the word to the stack.

    address: Obj,
    // Pushes the address to the stack.

    stack: usize,
    // Pushes a word from the stack to the stack. Offset 0 = top element,
    // 1 = below top, etc.

    pop: usize,
    // Pops the given number of words from the stack.

    popover: usize,
    // Keeps the top element on the stack, but pops the given number of words
    // below that.

    add, // a b -> (a+b)
    subtract, // a b -> (a-b)
    multiply, // a b -> (a*b)
    divide, // a b -> (a/b)
    modulo, // a b -> (a%b)
    shift_left, // a b -> (a<<b). Stands for shift left.
    shift_right, // a b -> (a>>b). Stands for shift right.
    and_, // a b -> (a & b). Bitwise and.
    or_, // a b -> (a | b). Bitwise or.
    xor, // a b -> (a ^ b). Bitwise xor.

    compare,
    // Pops two words. Stack before: a b.
    // If a == b, pushes 1. If a > b, pushes 1. If a < b, pushes 2.

    f_add, // a b -> (a+b), interpreted as f64.
    f_subtract, // a b -> (a-b), interpreted as f64.
    f_multiply, // a b -> (a*b), interpreted as f64.
    f_divide, // a b -> (a/b), interpreted as f64.

    f_compare,
    // Pops two f64s. Bit-equal -> 1, a>b -> 2, a<b -> 4, otherwise (NaN) -> 1.

    int_to_float, // Pops i64, pushes f64 with the same numeric value.
    float_to_int, // Pops f64, pushes i64 via truncation toward zero.
    f_is_finite, // Pops f64, pushes 1 if finite (not NaN, not +/-Inf), else 0.

    jump_if: usize,
    // Pops a word. If not 0, jumps to the instruction at the index.

    jump: usize,

    new_leaf: usize,
    // Creates a new heap object with the given number of words.

    new_inner: usize,
    // Creates a new heap object with the given number of words.

    flatten_to_inner,
    // Pops an address. An object of the form [a [b [c []]]] becomes [a b c].

    flatten_to_leaf,
    // Pops an address.
    // An object of the form [[a] [[b] [[c] []]]] becomes [a b c].

    points,
    // Pops an address. Returns 1 or 0, depending on whether the object contains
    // pointers.

    size, // Pops an address. Returns the size of the object.

    load, // addr offset -> ... Loads a word at the offset from the object.

    heap_size,
    // Pushes the size of the heap onto the stack. Can be used as a checkpoint
    // for gc.

    collect_garbage,
    // heapsize obj -> obj. Collects garbage starting at the heapsize, only
    // keeping dependencies of obj.

    call, // Pops an address, which should point to a function object. Calls it.

    call_indirect,
    // Pops fun_addr, pops inner heap object; pushes every child of the object;
    // runs fun.

    start_sandbox: Sandbox,
    // Pops a word (the fuel limit), opens a sandbox scope. The Sandbox changes
    // the behavior of the use_fuel and crash instructions before the matching
    // end_sandbox:
    // - On crash, we revert the call stack to the moment of start_sandbox. We
    //   set the instruction pointer to the Sandbox's on_crash. We revert the
    //   data stack to the moment of start_sandbox + the error.
    // - On out of fuel, we revert the call stack and data stack to the moment
    //   of start_sandbox. We set the instruction pointer to the Sandbox's
    //   on_out_of_fuel.

    end_sandbox, // Ends the current sandbox scope.

    use_fuel,
    // Pops a word (the fuel amount) and charges it against the current fuel
    // limit.

    crash,
    // Pops an error. Restores the surrounding sandbox's VM state. Pushes the
    // error. Jumps to the sandbox's on_crash handler.

    const Sandbox = struct { on_crash: usize, on_out_of_fuel: usize };

    pub fn format(instr: Instruction, writer: *std.io.Writer) !void {
        switch (instr) {
            .word => |word| try writer.print("word {}", .{word}),
            .address => |object| try writer.print("address {x}", .{object.address}),
            .stack => |offset| try writer.print("stack {}", .{offset}),
            .pop => |amount| try writer.print("pop {}", .{amount}),
            .popover => |amount| try writer.print("popover {}", .{amount}),
            .and_ => try writer.print("and", .{}),
            .or_ => try writer.print("or", .{}),
            .jump_if => |target| try writer.print("jump_if {}", .{target}),
            .jump => |target| try writer.print("jump {}", .{target}),
            .new_leaf => |size| try writer.print("new_leaf {}", .{size}),
            .new_inner => |size| try writer.print("new_inner {}", .{size}),
            .start_sandbox => |sandbox| try writer.print(
                "start_sandbox crash->{} out-of-fuel->",
                .{ sandbox.on_crash, sandbox.on_out_of_fuel },
            ),
            inline else => try writer.print("{s}", .{@tagName(instr)}),
        }
    }
};

pub fn init(heap: *Heap, ally: Ally) !Self {
    return .{
        .ally = ally,
        .heap = heap,
        .data_stack = try Stack.init(ally, 1000000),
        .compiled_cache = ObjMap(CompiledFun).empty,
    };
}

pub fn call(self: *Self, fun: Vm.Fun, args: []const Obj, fuel: *usize) !Vm.Result {
    // The byte code interpreter never returns error.UndefinedBehavior or
    // error.OutOfExpressions; both are in the signature only to match the
    // impl interface in vm.zig. (UB is detected by the tree-walker;
    // OutOfExpressions is its own host-side stack-guard.)
    //
    // Reset per-call state. The compiled_cache persists between runs.
    self.data_stack.used = 0;
    self.call_stack.items.len = 0;
    self.sandbox_stack.items.len = 0;
    self.crash_payload = null;
    self.fuel = fuel.*;
    defer fuel.* = self.fuel;
    errdefer fuel.* = self.fuel;

    const compiled = self.compile(fun) catch |err| switch (err) {
        error.OutOfMemory => return .out_of_memory,
        error.UndefinedBehavior => return error.UndefinedBehavior,
    };
    for (args) |arg| self.data_stack.push_obj(arg) catch return .out_of_memory;
    self.run(compiled.instructions) catch |err| switch (err) {
        error.OutOfMemory => return .out_of_memory,
        error.UncaughtCrash => return .{ .crashed = self.crash_payload.? },
        error.UncaughtOutOfFuel => {
            self.fuel = 0;
            return .out_of_fuel;
        },
        error.UndefinedBehavior => return error.UndefinedBehavior,
    };
    return .{ .returned = self.data_stack.pop_obj() };
}

pub fn compile(self: *Self, fun: Vm.Fun) !CompiledFun {
    if (self.compiled_cache.get(fun.obj)) |compiled| return compiled;
    const compiled = try really_compile(self.ally, fun);
    try self.compiled_cache.put(self.ally, fun.obj, compiled);
    return compiled;
}

pub fn really_compile(ally: Ally, fun: Vm.Fun) !CompiledFun {
    // std.debug.print("{f}", .{fun});
    var stack = std.ArrayList([]const u8).empty;
    defer stack.deinit(ally);
    for (fun.args()) |arg| try stack.append(ally, get_symbol(arg));
    var instrs = std.ArrayList(Instruction).empty;
    errdefer instrs.deinit(ally);
    try compile_expr(ally, fun, fun.body(), &stack, &instrs);
    if (fun.args().len > 0) try instrs.append(ally, .{ .popover = fun.args().len });
    const compiled = CompiledFun{ .instructions = try instrs.toOwnedSlice(ally) };
    // std.debug.print("{f}\n", .{compiled});
    return compiled;
}
fn find_in_stack(stack: *const std.ArrayList([]const u8), name: []const u8) ?usize {
    var i: usize = stack.items.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, stack.items[i], name)) return stack.items.len - 1 - i;
    }
    return null;
}
fn compile_expr(ally: std.mem.Allocator, root: Vm.Fun, expr: Vm.Expr, stack: *std.ArrayList([]const u8), instrs: *std.ArrayList(Instruction)) (Ally.Error || error{UndefinedBehavior})!void {
    const stack_size_before = stack.items.len;
    const k = try expr.kind();
    switch (k) {
        .word => |w| {
            try instrs.append(ally, .{ .word = w });
            try stack.append(ally, "");
        },
        .object => |obj| {
            try instrs.append(ally, .{ .address = obj });
            try stack.append(ally, "");
        },
        .name => |n| {
            if (find_in_stack(stack, n)) |offset| {
                try instrs.append(ally, .{ .stack = offset });
                try stack.append(ally, n);
            } else {
                return error.UndefinedBehavior;
            }
        },
        .let => |let| {
            try compile_expr(ally, root, let.def, stack, instrs);
            _ = stack.pop();
            try stack.append(ally, let.name);
            try compile_expr(ally, root, let.body, stack, instrs);
            try instrs.append(ally, .{ .popover = 1 });
            _ = stack.pop();
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .add, .subtract, .multiply, .divide, .modulo, .shift_left, .shift_right, .and_, .or_, .xor, .compare, .f_add, .f_subtract, .f_multiply, .f_divide, .f_compare => |args| {
            try compile_expr(ally, root, args.left, stack, instrs);
            try compile_expr(ally, root, args.right, stack, instrs);
            try instrs.append(ally, switch (k) {
                .add => .add,
                .subtract => .subtract,
                .multiply => .multiply,
                .divide => .divide,
                .modulo => .modulo,
                .shift_left => .shift_left,
                .shift_right => .shift_right,
                .and_ => .and_,
                .or_ => .or_,
                .xor => .xor,
                .compare => .compare,
                .f_add => .f_add,
                .f_subtract => .f_subtract,
                .f_multiply => .f_multiply,
                .f_divide => .f_divide,
                .f_compare => .f_compare,
                else => unreachable,
            });
            _ = stack.pop();
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .if_ => |if_| {
            try compile_expr(ally, root, if_.condition, stack, instrs);
            const after_condition = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder jump_if
            _ = stack.pop();
            try compile_expr(ally, root, if_.else_, stack, instrs);
            const after_else = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder jump
            const before_then = instrs.items.len;
            _ = stack.pop();
            try compile_expr(ally, root, if_.then, stack, instrs);
            const after_if = instrs.items.len;
            instrs.items[after_condition] = .{ .jump_if = before_then };
            instrs.items[after_else] = .{ .jump = after_if };
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .new_leaf => |args| {
            for (args) |arg| {
                try compile_expr(ally, root, arg, stack, instrs);
            }
            try instrs.append(ally, .{ .new_leaf = args.len });
            for (args) |_| _ = stack.pop();
            try stack.append(ally, "");
        },
        .new_inner => |args| {
            for (args) |arg| {
                try compile_expr(ally, root, arg, stack, instrs);
            }
            try instrs.append(ally, .{ .new_inner = args.len });
            for (args) |_| _ = stack.pop();
            try stack.append(ally, "");
        },
        .flatten_to_leaf, .flatten_to_inner, .points, .size, .crash, .int_to_float, .float_to_int, .f_is_finite => |inner| {
            try compile_expr(ally, root, inner, stack, instrs);
            try instrs.append(ally, switch (k) {
                .flatten_to_leaf => .flatten_to_leaf,
                .flatten_to_inner => .flatten_to_inner,
                .points => .points,
                .size => .size,
                .crash => .crash,
                .int_to_float => .int_to_float,
                .float_to_int => .float_to_int,
                .f_is_finite => .f_is_finite,
                else => unreachable,
            });
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .load => |load| {
            try compile_expr(ally, root, load.object, stack, instrs);
            try compile_expr(ally, root, load.index, stack, instrs);
            try instrs.append(ally, .load);
            _ = stack.pop();
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .collect_garbage => |inner| {
            try instrs.append(ally, .heap_size);
            try stack.append(ally, "");
            try compile_expr(ally, root, inner, stack, instrs);
            try instrs.append(ally, .collect_garbage);
            _ = stack.pop();
        },
        .sandbox => |sandbox| {
            try compile_expr(ally, root, sandbox.fuel, stack, instrs);
            const start = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder for start_sandbox
            _ = stack.pop();
            try compile_expr(ally, root, sandbox.body, stack, instrs);
            // Body ran through successfully.
            try instrs.append(ally, .end_sandbox);
            try instrs.append(ally, .{ .word = 0 });
            try instrs.append(ally, .{ .new_leaf = 1 });
            try instrs.append(ally, .{ .stack = 1 });
            try instrs.append(ally, .{ .new_inner = 2 });
            try instrs.append(ally, .{ .popover = 1 });
            const end_of_success = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder for jump
            // Body crashed.
            const on_crash = instrs.items.len;
            try instrs.append(ally, .{ .word = 1 });
            try instrs.append(ally, .{ .new_leaf = 1 });
            try instrs.append(ally, .{ .stack = 1 });
            try instrs.append(ally, .{ .new_inner = 2 });
            try instrs.append(ally, .{ .popover = 1 });
            const end_of_on_crash = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder for jump
            // Body out of fuel.
            const on_out_of_fuel = instrs.items.len;
            try instrs.append(ally, .{ .word = 2 });
            try instrs.append(ally, .{ .new_leaf = 1 });
            try instrs.append(ally, .{ .new_inner = 1 });
            const end_of_on_out_of_fuel = instrs.items.len;
            try instrs.append(ally, .crash); // placeholder for jump
            // Patch.
            const after_sandbox = instrs.items.len;
            instrs.items[start] = .{ .start_sandbox = .{
                .on_crash = on_crash,
                .on_out_of_fuel = on_out_of_fuel,
            } };
            instrs.items[end_of_success] = .{ .jump = after_sandbox };
            instrs.items[end_of_on_crash] = .{ .jump = after_sandbox };
            instrs.items[end_of_on_out_of_fuel] = .{ .jump = after_sandbox };
        },
        .use_fuel => |use_fuel| {
            try compile_expr(ally, root, use_fuel.amount, stack, instrs);
            try instrs.append(ally, .use_fuel);
            _ = stack.pop();
            try compile_expr(ally, root, use_fuel.child, stack, instrs);
        },
        .also => |also| {
            try compile_expr(ally, root, also.ignored, stack, instrs);
            try instrs.append(ally, .{ .pop = 1 });
            _ = stack.pop();
            try compile_expr(ally, root, also.value, stack, instrs);
        },
        .call => |c| {
            for (c.args) |arg| {
                try compile_expr(ally, root, arg, stack, instrs);
            }
            try compile_expr(ally, root, c.fun, stack, instrs);
            try instrs.append(ally, .call);
            _ = stack.pop();
            for (c.args) |_| _ = stack.pop();
            try stack.append(ally, "");
        },
        .call_indirect => |c| {
            try compile_expr(ally, root, c.args, stack, instrs);
            try compile_expr(ally, root, c.fun, stack, instrs);
            try instrs.append(ally, .call_indirect);
            _ = stack.pop();
            _ = stack.pop();
            try stack.append(ally, "");
        },
        .rec => |args| {
            for (args) |arg| {
                try compile_expr(ally, root, arg, stack, instrs);
            }
            try instrs.append(ally, .{ .address = root.obj });
            try instrs.append(ally, .call);
            for (args) |_| _ = stack.pop();
            try stack.append(ally, "");
        },
        .unreachable_ => {},
    }
    const stack_size_after = stack.items.len;
    std.debug.assert(stack_size_after == stack_size_before + 1);
}

fn run(self: *Self, instructions: []const Instruction) !void {
    var ip = Ip{ .instructions = instructions, .index = 0 };
    while (true) {
        if (ip.index >= ip.instructions.len) {
            ip = self.call_stack.pop() orelse return;
            continue;
        }
        const instruction = ip.instructions[ip.index];
        ip.index += 1;
        self.instruction_count += 1;
        switch (instruction) {
            .word => |word| try self.data_stack.push(word),
            .address => |object| try self.data_stack.push_obj(object),
            .stack => |offset| try self.data_stack.push(self.data_stack.get(offset)),
            .pop => |amount| self.data_stack.pop_n(amount),
            .popover => |amount| {
                const top = self.data_stack.pop();
                self.data_stack.pop_n(amount);
                try self.data_stack.push(top);
            },
            .add => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(a +% b);
            },
            .subtract => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(a -% b);
            },
            .multiply => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(a *% b);
            },
            .divide => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(@divTrunc(a, b));
            },
            .modulo => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(@mod(a, b));
            },
            .shift_left => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(a << @intCast(b));
            },
            .shift_right => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push_int(a >> @intCast(b));
            },
            .compare => {
                const b = self.data_stack.pop_int();
                const a = self.data_stack.pop_int();
                try self.data_stack.push(if (a == b) 1 else if (a > b) 2 else 4);
            },
            .f_add => {
                const b = self.data_stack.pop_float();
                const a = self.data_stack.pop_float();
                try self.data_stack.push_float(a + b);
            },
            .f_subtract => {
                const b = self.data_stack.pop_float();
                const a = self.data_stack.pop_float();
                try self.data_stack.push_float(a - b);
            },
            .f_multiply => {
                const b = self.data_stack.pop_float();
                const a = self.data_stack.pop_float();
                try self.data_stack.push_float(a * b);
            },
            .f_divide => {
                const b = self.data_stack.pop_float();
                const a = self.data_stack.pop_float();
                try self.data_stack.push_float(a / b);
            },
            .f_compare => {
                const b_bits = self.data_stack.pop();
                const a_bits = self.data_stack.pop();
                const result: Word = if (a_bits == b_bits) 1 else blk: {
                    const a: f64 = @bitCast(a_bits);
                    const b: f64 = @bitCast(b_bits);
                    break :blk if (a > b) 2 else if (a < b) 4 else 1;
                };
                try self.data_stack.push(result);
            },
            .int_to_float => try self.data_stack.push_float(@floatFromInt(self.data_stack.pop_int())),
            .float_to_int => try self.data_stack.push_int(@intFromFloat(self.data_stack.pop_float())),
            .f_is_finite => try self.data_stack.push(if (std.math.isFinite(self.data_stack.pop_float())) 1 else 0),
            .and_ => {
                const b = self.data_stack.pop();
                const a = self.data_stack.pop();
                try self.data_stack.push(a & b);
            },
            .or_ => {
                const b = self.data_stack.pop();
                const a = self.data_stack.pop();
                try self.data_stack.push(a | b);
            },
            .xor => {
                const b = self.data_stack.pop();
                const a = self.data_stack.pop();
                try self.data_stack.push(a ^ b);
            },
            .jump_if => |target| {
                const condition = self.data_stack.pop();
                if (condition != 0) ip.index = target;
            },
            .jump => |target| ip.index = target,
            .new_leaf => |size| {
                const stack = self.data_stack.memory[0..self.data_stack.used];
                const words = stack[stack.len - size ..];
                const obj = try self.heap.new_leaf(@ptrCast(words));
                self.data_stack.pop_n(size);
                try self.data_stack.push_obj(obj);
            },
            .new_inner => |size| {
                const stack = self.data_stack.memory[0..self.data_stack.used];
                const words = stack[stack.len - size ..];
                const obj = try self.heap.new_inner(@ptrCast(words));
                self.data_stack.pop_n(size);
                try self.data_stack.push_obj(obj);
            },
            .flatten_to_inner => {
                var obj = self.data_stack.pop_obj();
                var b = try self.heap.build_inner();
                while (true) {
                    switch (obj.size()) {
                        0 => break,
                        2 => {
                            try b.emit(obj.child(0));
                            obj = obj.child(1);
                        },
                        else => unreachable,
                    }
                }
                try self.data_stack.push_obj(b.finish());
            },
            .flatten_to_leaf => {
                var obj = self.data_stack.pop_obj();
                var b = try self.heap.build_leaf();
                while (true) {
                    switch (obj.size()) {
                        0 => break,
                        2 => {
                            try b.emit(obj.child(0).word(0));
                            obj = obj.child(1);
                        },
                        else => unreachable,
                    }
                }
                try self.data_stack.push_obj(b.finish());
            },
            .points => {
                const obj = self.data_stack.pop_obj();
                try self.data_stack.push(if (obj.is_inner()) 1 else 0);
            },
            .size => {
                const obj = self.data_stack.pop_obj();
                try self.data_stack.push(obj.size());
            },
            .load => {
                const offset: usize = self.data_stack.pop();
                const base = self.data_stack.pop_obj();
                const word: usize = if (base.is_inner()) base.child(offset).address else base.word(offset);
                try self.data_stack.push(word);
            },
            .heap_size => {
                const checkpoint = self.heap.checkpoint();
                try self.data_stack.push(checkpoint.address);
            },
            .collect_garbage => {
                const keep = self.data_stack.pop_obj();
                const checkpoint = self.data_stack.pop();
                self.heap.dump_stats();
                std.debug.print("garbage collecting...\n", .{});
                const mapped_keep = try self.garbage_collect(
                    .{ .address = checkpoint },
                    keep,
                );
                self.heap.dump_stats();
                try self.data_stack.push_obj(mapped_keep);
            },
            .call => {
                const callee = self.data_stack.pop_obj();
                const compiled = try self.compile(Vm.Fun{ .obj = callee });
                try self.call_stack.append(self.ally, ip);
                ip = .{ .instructions = compiled.instructions, .index = 0 };
            },
            .call_indirect => {
                const callee = self.data_stack.pop_obj();
                const args = self.data_stack.pop_obj();
                for (args.children()) |c| try self.data_stack.push_obj(c);
                const compiled = try self.compile(Vm.Fun{ .obj = callee });
                try self.call_stack.append(self.ally, ip);
                ip = .{ .instructions = compiled.instructions, .index = 0 };
            },
            .start_sandbox => |sandbox| {
                const limit: usize = @intCast(self.data_stack.pop_int());
                const fuel_diff = if (self.fuel <= limit) 0 else self.fuel - limit;
                self.fuel -= fuel_diff;
                try self.sandbox_stack.append(self.ally, .{
                    .data_stack_size = self.data_stack.used,
                    .call_stack_size = self.call_stack.items.len,
                    .heap_checkpoint = self.heap.checkpoint(),
                    .fuel_diff = fuel_diff,
                    .on_crash = .{ .instructions = ip.instructions, .index = sandbox.on_crash },
                    .on_out_of_fuel = .{ .instructions = ip.instructions, .index = sandbox.on_out_of_fuel },
                });
            },
            .end_sandbox => {
                const scope = self.sandbox_stack.pop().?;
                self.fuel += scope.fuel_diff;
            },
            .use_fuel => {
                const amount: usize = @intCast(self.data_stack.pop_int());
                if (self.fuel < amount) {
                    @branchHint(.unlikely);
                    const sandbox = sandbox: while (true) {
                        const sandbox = self.sandbox_stack.pop() orelse {
                            return error.UncaughtOutOfFuel;
                        };
                        if (sandbox.fuel_diff == 0) continue;
                        break :sandbox sandbox;
                    };
                    self.data_stack.used = sandbox.data_stack_size;
                    self.call_stack.items.len = sandbox.call_stack_size;
                    self.fuel = sandbox.fuel_diff;
                    ip = sandbox.on_out_of_fuel;
                    self.heap.restore(sandbox.heap_checkpoint);
                    self.compiled_cache.remove_everything_after(sandbox.heap_checkpoint.address);
                } else self.fuel -= amount;
            },
            .crash => {
                @branchHint(.unlikely);
                const sandbox = self.sandbox_stack.pop() orelse {
                    self.crash_payload = self.data_stack.pop_obj();
                    return error.UncaughtCrash;
                };
                const e = self.data_stack.pop_obj();
                self.data_stack.used = sandbox.data_stack_size;
                self.call_stack.items.len = sandbox.call_stack_size;
                self.fuel += sandbox.fuel_diff;
                ip = sandbox.on_crash;
                const mapped_e = try self.garbage_collect(sandbox.heap_checkpoint, e);
                try self.data_stack.push_obj(mapped_e);
            },
        }
    }
}

pub fn deduplicate(self: *Self, checkpoint: Heap.Checkpoint, obj: Obj) !Obj {
    var map = try self.heap.deduplicate(self.ally, checkpoint);
    defer map.deinit();
    self.compiled_cache.remove_everything_after(checkpoint.address);
    return map.get(obj) orelse obj;
}

pub fn garbage_collect(self: *Self, checkpoint: Heap.Checkpoint, keep: Obj) !Obj {
    const result = try self.heap.garbage_collect(self.ally, checkpoint, keep);
    self.compiled_cache.remove_everything_after(result.stable_until);
    return result.keep;
}
