// A tree-walking interpreter for the VM's IR.
//
// This is an alternative to vm_byte_code.zig and vm_x86_64.zig, designed for
// correctness rather than performance. It walks the IR (Vm.Expr) directly using
// Zig's call stack, with no instruction compilation, no data stack, and no call
// stack of its own.
//
// The whole point is to catch undefined behavior. Anywhere bootstrap.objects
// says "Assumes that ...", this interpreter validates the assumption and bails
// with error.UndefinedBehavior when it's violated. The other VM implementations
// exploit undefined behavior for performance; this one acts as a strict
// reference that can be used to find bugs in code that compiles and runs but
// is actually invalid.
//
// Values are tracked as a tagged union of word-or-object. Objects are treated
// as opaque identifiers: doing arithmetic on them, comparing them, or trying to
// use them as a count or condition is rejected. By construction, every Object
// value in the interpreter is the address of an actual heap object — there is
// no per-use validation needed.
//
// Fuel, crashes, and sandboxes are threaded through the recursion explicitly:
// each eval returns an ExprResult that's one of returned/crashed/out-of-fuel/
// out-of-memory. There is no global VM state for any of it. The fuel pointer
// is decremented in place by use-fuel; sandboxes cap it for their body and
// restore it afterward.

const std = @import("std");
const Ally = std.mem.Allocator;

const Heap = @import("heap.zig");
const Word = Heap.Word;
const Obj = Heap.Obj;
const Vm = @import("vm.zig");
const get_symbol = Heap.get_symbol;

const Self = @This();

ally: Ally,
heap: *Heap,
log_ub: bool,
max_expressions: usize,
expressions_evaluated: usize = 0,

const Options = struct {
    log_undefined_behavior: bool = true,
    max_expressions: usize = std.math.maxInt(usize),
};
pub fn init(heap: *Heap, ally: Ally, options: Options) !Self {
    return .{
        .ally = ally,
        .heap = heap,
        .log_ub = options.log_undefined_behavior,
        .max_expressions = options.max_expressions,
    };
}

pub const Value = union(enum) {
    word: Word,
    object: Obj,

    pub fn format(v: Value, writer: *std.Io.Writer) !void {
        switch (v) {
            .word => |w| try writer.print("word {d}", .{w}),
            .object => |o| try writer.print("object {x}", .{o.address}),
        }
    }
};
pub const ExprResult = union(enum) {
    returned: Value,
    crashed: Obj,
    out_of_fuel,
    out_of_memory,

    fn ok(self: ExprResult) ?Value {
        return switch (self) {
            .returned => |v| v,
            else => null,
        };
    }
};

const Binding = struct { name: []const u8, value: Value };
const Env = struct {
    bindings: std.ArrayList(Binding) = .empty,

    fn deinit(env: *Env, ally: Ally) void {
        env.bindings.deinit(ally);
    }
    fn push(env: *Env, ally: Ally, name: []const u8, value: Value) !void {
        try env.bindings.append(ally, .{ .name = name, .value = value });
    }
    fn pop(env: *Env) void {
        _ = env.bindings.pop();
    }
    fn lookup(env: Env, name: []const u8) ?Value {
        var i = env.bindings.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, env.bindings.items[i].name, name)) {
                return env.bindings.items[i].value;
            }
        }
        return null;
    }
};

pub fn call(self: *Self, fun: Vm.Fun, args: []const Obj, fuel: *usize) error{ UndefinedBehavior, OutOfExpressions }!Vm.Result {
    self.expressions_evaluated = 0;

    const arg_values = self.ally.alloc(Value, args.len) catch return .out_of_memory;
    defer self.ally.free(arg_values);
    for (args, 0..) |o, i| arg_values[i] = .{ .object = o };

    const result = try self.eval_call(fun, arg_values, fuel);
    return switch (result) {
        .returned => |v| switch (v) {
            .object => |o| .{ .returned = o },
            .word => self.fail("function returned non- object", .{}),
        },
        .crashed => |e| .{ .crashed = e },
        .out_of_fuel => .out_of_fuel,
        .out_of_memory => .out_of_memory,
    };
}

pub fn garbage_collect(self: *Self, checkpoint: Heap.Checkpoint, keep: Obj) !Obj {
    return (try self.heap.garbage_collect(self.ally, checkpoint, keep)).keep;
}

pub fn deduplicate(self: *Self, checkpoint: Heap.Checkpoint, obj: Obj) !Obj {
    var map = try self.heap.deduplicate(self.ally, checkpoint);
    defer map.deinit();
    return map.get(obj) orelse obj;
}

fn fail(self: *Self, comptime fmt: []const u8, args: anytype) error{UndefinedBehavior} {
    if (self.log_ub) {
        std.debug.print("\nCaught undefined behavior:\n  " ++ fmt ++ "\n", args);
    }
    return error.UndefinedBehavior;
}
fn expect_word(self: *Self, v: Value, comptime context: []const u8) error{UndefinedBehavior}!Word {
    return switch (v) {
        .word => |w| w,
        .object => self.fail(context ++ " expected a word, got an object.", .{}),
    };
}
fn expect_object(self: *Self, v: Value, comptime context: []const u8) error{UndefinedBehavior}!Obj {
    return switch (v) {
        .object => |o| o,
        .word => self.fail(context ++ " expected an object, got a word.", .{}),
    };
}
fn validate_shift_amount(self: *Self, b: i64) error{UndefinedBehavior}!u6 {
    if (b < 0 or b >= 64) return self.fail("Shift amount {d} is out of range [0, 64).", .{b});
    return @intCast(b);
}

fn validate_function_shape(self: *Self, obj: Obj) error{UndefinedBehavior}!Vm.Fun {
    if (!obj.is_inner()) return self.fail("Call target is not an inner object.", .{});
    if (obj.size() != 2) {
        return self.fail("Call target has {d} children, expected 2 (args, body).", .{obj.size()});
    }
    const args = obj.child(0);
    if (!args.is_inner()) return self.fail("Call target's argument list is not an inner object.", .{});
    for (args.children()) |arg_name| {
        if (!arg_name.is_leaf()) return self.fail("Function argument name is not a leaf object.", .{});
    }
    const body = obj.child(1);
    if (!body.is_inner()) return self.fail("Function body is not an inner object.", .{});
    if (body.size() == 0) return self.fail("Function body has no tag.", .{});
    return Vm.Fun{ .obj = obj };
}

fn eval_call(self: *Self, fun: Vm.Fun, args: []const Value, fuel: *usize) error{ UndefinedBehavior, OutOfExpressions }!ExprResult {
    _ = try self.validate_function_shape(fun.obj);
    if (fun.args().len != args.len) {
        return self.fail("Arity mismatch: function takes a different number of arguments.", .{});
    }
    var env = Env{};
    defer env.deinit(self.ally);
    for (fun.args(), args) |arg_name, value| {
        env.push(self.ally, get_symbol(arg_name), value) catch return .out_of_memory;
    }
    return self.eval(fun.body(), &env, fun, fuel);
}

fn eval(self: *Self, expr: Vm.Expr, env: *Env, current_fun: Vm.Fun, fuel: *usize) error{ UndefinedBehavior, OutOfExpressions }!ExprResult {
    if (self.expressions_evaluated >= self.max_expressions) return error.OutOfExpressions;
    self.expressions_evaluated += 1;
    switch (try expr.kind()) {
        .word => |w| return .{ .returned = .{ .word = w } },
        .object => |obj| return .{ .returned = .{ .object = obj } },
        .name => |name| {
            const v = env.lookup(name) orelse return self.fail("Unbound name: {s}", .{name});
            return .{ .returned = v };
        },
        .let => |let| {
            const def_result = try self.eval(let.def, env, current_fun, fuel);
            const def = def_result.ok() orelse return def_result;
            env.push(self.ally, let.name, def) catch return .out_of_memory;
            defer env.pop();
            return self.eval(let.body, env, current_fun, fuel);
        },
        .also => |also| {
            const ignored_result = try self.eval(also.ignored, env, current_fun, fuel);
            _ = ignored_result.ok() orelse return ignored_result;
            return self.eval(also.value, env, current_fun, fuel);
        },
        .add => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "add left"));
            const r: i64 = @bitCast(try self.expect_word(right, "add right"));
            return .{ .returned = .{ .word = @bitCast(l +% r) } };
        },
        .subtract => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "subtract left"));
            const r: i64 = @bitCast(try self.expect_word(right, "subtract right"));
            return .{ .returned = .{ .word = @bitCast(l -% r) } };
        },
        .multiply => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "multiply left"));
            const r: i64 = @bitCast(try self.expect_word(right, "multiply right"));
            return .{ .returned = .{ .word = @bitCast(l *% r) } };
        },
        .divide => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "divide left"));
            const r: i64 = @bitCast(try self.expect_word(right, "divide right"));
            if (r == 0) return self.fail("Division by zero.", .{});
            if (l == std.math.minInt(i64) and r == -1) {
                return self.fail("Division of minInt(i64) by -1 overflows.", .{});
            }
            return .{ .returned = .{ .word = @bitCast(@divTrunc(l, r)) } };
        },
        .modulo => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "modulo left"));
            const r: i64 = @bitCast(try self.expect_word(right, "modulo right"));
            if (r == 0) return self.fail("Modulo by zero.", .{});
            if (l == std.math.minInt(i64) and r == -1) {
                return self.fail("Modulo of minInt(i64) by -1 overflows.", .{});
            }
            return .{ .returned = .{ .word = @bitCast(@mod(l, r)) } };
        },
        .shift_left => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l = try self.expect_word(left, "shift-left left");
            const r: i64 = @bitCast(try self.expect_word(right, "shift-left right"));
            return .{ .returned = .{ .word = l << try self.validate_shift_amount(r) } };
        },
        .shift_right => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l = try self.expect_word(left, "shift-right left");
            const r: i64 = @bitCast(try self.expect_word(right, "shift-right right"));
            return .{ .returned = .{ .word = l >> try self.validate_shift_amount(r) } };
        },
        .and_ => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l = try self.expect_word(left, "and left");
            const r = try self.expect_word(right, "and right");
            return .{ .returned = .{ .word = l & r } };
        },
        .or_ => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l = try self.expect_word(left, "or left");
            const r = try self.expect_word(right, "or right");
            return .{ .returned = .{ .word = l | r } };
        },
        .xor => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l = try self.expect_word(left, "xor left");
            const r = try self.expect_word(right, "xor right");
            return .{ .returned = .{ .word = l ^ r } };
        },
        .compare => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: i64 = @bitCast(try self.expect_word(left, "compare left"));
            const r: i64 = @bitCast(try self.expect_word(right, "compare right"));
            return .{ .returned = .{ .word = if (l == r) 1 else if (l > r) 2 else 4 } };
        },
        .f_add => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: f64 = @bitCast(try self.expect_word(left, "f-add left"));
            const r: f64 = @bitCast(try self.expect_word(right, "f-add right"));
            return .{ .returned = .{ .word = @bitCast(l + r) } };
        },
        .f_subtract => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: f64 = @bitCast(try self.expect_word(left, "f-subtract left"));
            const r: f64 = @bitCast(try self.expect_word(right, "f-subtract right"));
            return .{ .returned = .{ .word = @bitCast(l - r) } };
        },
        .f_multiply => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: f64 = @bitCast(try self.expect_word(left, "f-multiply left"));
            const r: f64 = @bitCast(try self.expect_word(right, "f-multiply right"));
            return .{ .returned = .{ .word = @bitCast(l * r) } };
        },
        .f_divide => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l: f64 = @bitCast(try self.expect_word(left, "f-divide left"));
            const r: f64 = @bitCast(try self.expect_word(right, "f-divide right"));
            return .{ .returned = .{ .word = @bitCast(l / r) } };
        },
        .f_compare => |args| {
            const left_result = try self.eval(args.left, env, current_fun, fuel);
            const left = left_result.ok() orelse return left_result;
            const right_result = try self.eval(args.right, env, current_fun, fuel);
            const right = right_result.ok() orelse return right_result;
            const l_bits = try self.expect_word(left, "f-compare left");
            const r_bits = try self.expect_word(right, "f-compare right");
            if (l_bits == r_bits) return .{ .returned = .{ .word = 1 } };
            const l: f64 = @bitCast(l_bits);
            const r: f64 = @bitCast(r_bits);
            return .{ .returned = .{ .word = if (l > r) 2 else if (l < r) 4 else 1 } };
        },
        .int_to_float => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const i: i64 = @bitCast(try self.expect_word(v, "int-to-float"));
            return .{ .returned = .{ .word = @bitCast(@as(f64, @floatFromInt(i))) } };
        },
        .float_to_int => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const f: f64 = @bitCast(try self.expect_word(v, "float-to-int"));
            if (!std.math.isFinite(f)) return self.fail("float-to-int on non-finite value.", .{});
            const min: f64 = @floatFromInt(std.math.minInt(i64));
            const max: f64 = @floatFromInt(std.math.maxInt(i64));
            if (f < min or f > max) {
                return self.fail("float-to-int conversion out of i64 range: {d}", .{f});
            }
            return .{ .returned = .{ .word = @bitCast(@as(i64, @intFromFloat(f))) } };
        },
        .f_is_finite => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const f: f64 = @bitCast(try self.expect_word(v, "f-is-finite"));
            return .{ .returned = .{ .word = if (std.math.isFinite(f)) 1 else 0 } };
        },
        .if_ => |if_| {
            const cond_result = try self.eval(if_.condition, env, current_fun, fuel);
            const cond_v = cond_result.ok() orelse return cond_result;
            const cond = try self.expect_word(cond_v, "if condition");
            const branch = if (cond != 0) if_.then else if_.else_;
            return self.eval(branch, env, current_fun, fuel);
        },
        .new_leaf => |children| {
            const words = self.ally.alloc(Word, children.len) catch return .out_of_memory;
            defer self.ally.free(words);
            for (children, 0..) |c, i| {
                const r = try self.eval(c, env, current_fun, fuel);
                const v = r.ok() orelse return r;
                words[i] = try self.expect_word(v, "new-leaf child");
            }
            const obj = self.heap.new_leaf(words) catch return .out_of_memory;
            return .{ .returned = .{ .object = obj } };
        },
        .new_inner => |children| {
            const objs = self.ally.alloc(Obj, children.len) catch return .out_of_memory;
            defer self.ally.free(objs);
            for (children, 0..) |c, i| {
                const r = try self.eval(c, env, current_fun, fuel);
                const v = r.ok() orelse return r;
                objs[i] = try self.expect_object(v, "new-inner child");
            }
            const obj = self.heap.new_inner(objs) catch return .out_of_memory;
            return .{ .returned = .{ .object = obj } };
        },
        .flatten_to_leaf => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            var list = try self.expect_object(v, "flatten-to-leaf");
            var b = self.heap.build_leaf() catch return .out_of_memory;
            while (true) {
                if (!list.is_inner()) {
                    return self.fail("flatten-to-leaf encountered a leaf as a list cell.", .{});
                }
                switch (list.size()) {
                    0 => break,
                    2 => {
                        const item = list.child(0);
                        if (!item.is_leaf()) {
                            return self.fail("flatten-to-leaf item is not a leaf object.", .{});
                        }
                        if (item.size() != 1) {
                            return self.fail(
                                "flatten-to-leaf item is a leaf with {d} words, expected 1.",
                                .{item.size()},
                            );
                        }
                        b.emit(item.word(0)) catch return .out_of_memory;
                        list = list.child(1);
                    },
                    else => return self.fail(
                        "flatten-to-leaf list cell has {d} children, expected 0 or 2.",
                        .{list.size()},
                    ),
                }
            }
            return .{ .returned = .{ .object = b.finish() } };
        },
        .flatten_to_inner => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            var list = try self.expect_object(v, "flatten-to-inner");
            var b = self.heap.build_inner() catch return .out_of_memory;
            while (true) {
                if (!list.is_inner()) {
                    return self.fail("flatten-to-inner encountered a leaf as a list cell.", .{});
                }
                switch (list.size()) {
                    0 => break,
                    2 => {
                        b.emit(list.child(0)) catch return .out_of_memory;
                        list = list.child(1);
                    },
                    else => return self.fail(
                        "flatten-to-inner list cell has {d} children, expected 0 or 2.",
                        .{list.size()},
                    ),
                }
            }
            return .{ .returned = .{ .object = b.finish() } };
        },
        .points => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const obj = try self.expect_object(v, "points");
            return .{ .returned = .{ .word = if (obj.is_inner()) 1 else 0 } };
        },
        .size => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const obj = try self.expect_object(v, "size");
            return .{ .returned = .{ .word = obj.size() } };
        },
        .load => |load| {
            const obj_result = try self.eval(load.object, env, current_fun, fuel);
            const o_v = obj_result.ok() orelse return obj_result;
            const obj = try self.expect_object(o_v, "load object");
            const idx_result = try self.eval(load.index, env, current_fun, fuel);
            const i_v = idx_result.ok() orelse return idx_result;
            const idx_i: i64 = @bitCast(try self.expect_word(i_v, "load index"));
            if (idx_i < 0) return self.fail("load index {d} is negative.", .{idx_i});
            const idx: usize = @intCast(idx_i);
            if (idx >= obj.size()) {
                return self.fail("load index {d} is out of bounds (size {d}).", .{ idx, obj.size() });
            }
            return .{ .returned = if (obj.is_inner())
                .{ .object = obj.child(idx) }
            else
                .{ .word = obj.word(idx) } };
        },
        .collect_garbage => |arg| {
            const checkpoint = self.heap.checkpoint();
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            const keep = try self.expect_object(v, "gc");
            const mapped = self.garbage_collect(checkpoint, keep) catch return .out_of_memory;
            return .{ .returned = .{ .object = mapped } };
        },
        .call => |c| {
            const arg_values = self.ally.alloc(Value, c.args.len) catch return .out_of_memory;
            defer self.ally.free(arg_values);
            for (c.args, 0..) |a, i| {
                const r = try self.eval(a, env, current_fun, fuel);
                arg_values[i] = r.ok() orelse return r;
            }
            const fun_result = try self.eval(c.fun, env, current_fun, fuel);
            const fv = fun_result.ok() orelse return fun_result;
            const fun_obj = try self.expect_object(fv, "call function");
            return self.eval_call(.{ .obj = fun_obj }, arg_values, fuel);
        },
        .call_indirect => |ci| {
            const args_result = try self.eval(ci.args, env, current_fun, fuel);
            const av = args_result.ok() orelse return args_result;
            const args_obj = try self.expect_object(av, "call-indirect args");
            if (!args_obj.is_inner()) {
                return self.fail("call-indirect args is not an inner object.", .{});
            }
            const fun_result = try self.eval(ci.fun, env, current_fun, fuel);
            const fv = fun_result.ok() orelse return fun_result;
            const fun_obj = try self.expect_object(fv, "call-indirect function");
            const children = args_obj.children();
            const arg_values = self.ally.alloc(Value, children.len) catch return .out_of_memory;
            defer self.ally.free(arg_values);
            for (children, 0..) |c, i| arg_values[i] = .{ .object = c };
            return self.eval_call(.{ .obj = fun_obj }, arg_values, fuel);
        },
        .rec => |args| {
            const arg_values = self.ally.alloc(Value, args.len) catch return .out_of_memory;
            defer self.ally.free(arg_values);
            for (args, 0..) |a, i| {
                const r = try self.eval(a, env, current_fun, fuel);
                arg_values[i] = r.ok() orelse return r;
            }
            return self.eval_call(current_fun, arg_values, fuel);
        },
        .sandbox => |sb| {
            const fuel_result = try self.eval(sb.fuel, env, current_fun, fuel);
            const fv = fuel_result.ok() orelse return fuel_result;
            const limit_i: i64 = @bitCast(try self.expect_word(fv, "sandbox fuel"));
            if (limit_i < 0) return self.fail("sandbox fuel limit {d} is negative.", .{limit_i});
            const limit: usize = @intCast(limit_i);

            // If our cap is below the outer fuel, swap it in for the body and
            // remember the slack to restore afterward. Otherwise let the body
            // consume from the outer pool directly.
            const saved_fuel = fuel.*;
            const this_sandbox_caps = limit < saved_fuel;
            if (this_sandbox_caps) fuel.* = limit;

            const checkpoint = self.heap.checkpoint();
            const body = try self.eval(sb.body, env, current_fun, fuel);
            if (this_sandbox_caps) {
                // body's "remaining" is fuel.*; it consumed (limit - fuel.*).
                fuel.* = saved_fuel - limit + fuel.*;
            }

            switch (body) {
                .returned => |v| {
                    const result_obj = try self.expect_object(v, "sandbox body result");
                    const zero = self.heap.new_leaf(&.{0}) catch return .out_of_memory;
                    const wrapped = self.heap.new_inner(&.{ zero, result_obj }) catch return .out_of_memory;
                    return .{ .returned = .{ .object = wrapped } };
                },
                .crashed => |e| {
                    const mapped = self.garbage_collect(checkpoint, e) catch return .out_of_memory;
                    const one = self.heap.new_leaf(&.{1}) catch return .out_of_memory;
                    const wrapped = self.heap.new_inner(&.{ one, mapped }) catch return .out_of_memory;
                    return .{ .returned = .{ .object = wrapped } };
                },
                .out_of_fuel => {
                    if (this_sandbox_caps) {
                        // Our own cap ran out, but the outer pool still has
                        // fuel. Absorb the OOF: restore the heap and produce
                        // the [2] sentinel.
                        self.heap.restore(checkpoint);
                        const two = self.heap.new_leaf(&.{2}) catch return .out_of_memory;
                        const wrapped = self.heap.new_inner(&.{two}) catch return .out_of_memory;
                        return .{ .returned = .{ .object = wrapped } };
                    } else {
                        // We weren't restrictive, so the outer pool is the
                        // one that's exhausted; propagate. The absorbing
                        // sandbox above us will restore the heap.
                        return .out_of_fuel;
                    }
                },
                .out_of_memory => return .out_of_memory,
            }
        },
        .use_fuel => |uf| {
            const amount_result = try self.eval(uf.amount, env, current_fun, fuel);
            const av = amount_result.ok() orelse return amount_result;
            const amount_i: i64 = @bitCast(try self.expect_word(av, "use-fuel amount"));
            if (amount_i < 0) return self.fail("use-fuel amount {d} is negative.", .{amount_i});
            const amount: usize = @intCast(amount_i);
            if (amount > fuel.*) {
                fuel.* = 0;
                return .out_of_fuel;
            }
            fuel.* -= amount;
            return self.eval(uf.child, env, current_fun, fuel);
        },
        .crash => |arg| {
            const arg_result = try self.eval(arg, env, current_fun, fuel);
            const v = arg_result.ok() orelse return arg_result;
            return .{ .crashed = try self.expect_object(v, "crash") };
        },
        .unreachable_ => return self.fail("Reached an 'unreachable' expression.", .{}),
    }
}
