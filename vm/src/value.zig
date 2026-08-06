const std = @import("std");
const Heap = @import("heap.zig");
const Address = Heap.Address;
const Word = Heap.Word;
const Obj = Heap.Obj;
const Vm = @import("vm.zig");

obj: Obj,

const Value = @This();

pub fn from(obj: Obj) Value {
    return .{ .obj = obj };
}

// pub fn new_empty(heap: *Heap) !Obj {
//     var b = try heap.build_leaf();
//     return b.finish();
// }

pub const Kind = union(enum) { int, float, string, struct_, enum_, array, function };
pub fn kind(self: Value) Kind {
    const obj = self.obj;
    const kind_str = Heap.get_symbol(obj.child(0).child(0));
    if (std.mem.eql(u8, kind_str, "int")) return .int;
    if (std.mem.eql(u8, kind_str, "float")) return .float;
    if (std.mem.eql(u8, kind_str, "string")) return .string;
    if (std.mem.eql(u8, kind_str, "struct")) return .struct_;
    if (std.mem.eql(u8, kind_str, "enum")) return .enum_;
    if (std.mem.eql(u8, kind_str, "array")) return .array;
    if (std.mem.eql(u8, kind_str, "function")) return .function;
    std.debug.print("kind '{s}'\n", .{kind_str});
    @panic("unknown type ");
}

// Int stuff.

pub fn new_int(heap: *Heap, value: i64) !Value {
    const int_symbol = try heap.new_symbol("int");
    const type_ = try heap.new_inner(&.{int_symbol});
    const literal_obj = try heap.new_leaf(&.{@bitCast(value)});
    const int_obj = try heap.new_inner(&.{ type_, literal_obj });
    return .{ .obj = int_obj };
}

pub fn int(self: Value) i64 {
    std.debug.assert(self.kind() == .int);
    return @bitCast(self.obj.child(1).word(0));
}

// Float stuff.

pub fn new_float(heap: *Heap, value: f64) !Value {
    const float_symbol = try heap.new_symbol("float");
    const type_ = try heap.new_inner(&.{float_symbol});
    const literal_obj = try heap.new_leaf(&.{@bitCast(value)});
    const float_obj = try heap.new_inner(&.{ type_, literal_obj });
    return .{ .obj = float_obj };
}

pub fn float(self: Value) f64 {
    std.debug.assert(self.kind() == .float);
    return @bitCast(self.obj.child(1).word(0));
}

// String stuff.

pub fn new_string(heap: *Heap, string_: []const u8) !Value {
    const string_symbol = try heap.new_symbol("string");
    const ty = try heap.new_inner(&.{string_symbol});
    const symbol = try heap.new_symbol(string_);
    return .{ .obj = try heap.new_inner(&.{ ty, symbol }) };
}

pub fn string(self: Value) []const u8 {
    std.debug.assert(self.kind() == .string);
    return Heap.get_symbol(self.obj.child(1));
}

// Struct stuff.

pub fn new_struct(heap: *Heap, data: anytype) !Value {
    const struct_symbol = try heap.new_symbol("struct");
    const info = @typeInfo(@TypeOf(data));
    var keys: [info.@"struct".fields.len]Obj = undefined;
    inline for (info.@"struct".fields, 0..) |f, i| {
        keys[i] = try heap.new_symbol(f.name);
    }
    const ty = obj: {
        var b = try heap.build_inner();
        try b.emit(struct_symbol);
        for (keys) |k| try b.emit(k);
        break :obj b.finish();
    };
    const val = obj: {
        var b = try heap.build_inner();
        try b.emit(ty);
        inline for (info.@"struct".fields) |f| {
            try b.emit(@field(data, f.name).obj);
        }
        break :obj b.finish();
    };
    return Value{ .obj = val };
}

pub fn field(self: Value, name: []const u8) Value {
    std.debug.assert(self.kind() == .struct_);
    const ty = self.obj.child(0).children();
    for (ty[1..], 1..) |field_, i| {
        if (std.mem.eql(u8, Heap.get_symbol(field_), name)) {
            return Value.from(self.obj.child(i));
        }
    }
    @panic("field");
}

pub fn new_nil(heap: *Heap) !Value {
    return try Value.new_struct(heap, .{});
}

// Enum stuff.

pub fn new_enum(heap: *Heap, variant_: []const u8, payload_: Value) !Value {
    const enum_symbol = try heap.new_symbol("enum");
    const variant_symbol = try heap.new_symbol(variant_);
    const ty = try heap.new_inner(&.{ enum_symbol, variant_symbol });
    return Value.from(try heap.new_inner(&.{ ty, payload_.obj }));
}

pub fn variant(self: Value) []const u8 {
    std.debug.assert(self.kind() == .enum_);
    return Heap.get_symbol(self.obj.child(0).child(1));
}
pub fn payload(self: Value) Value {
    std.debug.assert(self.kind() == .enum_);
    return .{ .obj = self.obj.child(1) };
}

pub fn new_bool(heap: *Heap, b: bool) !Value {
    return try Value.new_enum(
        heap,
        if (b) "true" else "false",
        try Value.new_nil(heap),
    );
}
pub fn bool_(self: Value) bool {
    const variant_ = self.variant();
    if (std.mem.eql(u8, variant_, "true")) return true;
    if (std.mem.eql(u8, variant_, "false")) return false;
    @panic("not a bool");
}

pub fn new_option(heap: *Heap, option_: ?Value) !Value {
    if (option_) |payload_| {
        return try Value.new_enum(heap, "some", payload_);
    } else {
        return try Value.new_enum(heap, "none", try Value.new_nil(heap));
    }
}
pub fn option(self: Value) ?Value {
    const variant_ = self.variant();
    if (std.mem.eql(u8, variant_, "some")) return self.payload();
    if (std.mem.eql(u8, variant_, "none")) return null;
    @panic("not an option");
}

// Array stuff.

pub fn items(self: Value) []const Value {
    std.debug.assert(self.kind() == .array);
    return @ptrCast(self.obj.children()[1..]);
}

// Function stuff.

pub fn new_function(heap: *Heap, args: Obj, ir_: Obj, compiled: Obj, captured_: Obj) !Value {
    const function_symbol = try heap.new_symbol("function");
    const ty = obj: {
        var b = try heap.build_inner();
        try b.emit(function_symbol);
        try b.emit(args);
        try b.emit(ir_);
        try b.emit(compiled);
        break :obj b.finish();
    };
    return try heap.new_inner(&.{ ty, captured_ });
}
pub fn arguments(self: Value) Obj {
    return self.obj.child(0).child(1);
}
pub fn ir(self: Value) Obj {
    return self.obj.child(0).child(2);
}
pub fn vm_fun(self: Value) Vm.Fun {
    return Vm.Fun{ .obj = self.obj.child(0).child(3) };
}
pub fn captured(self: Value) Value {
    return Value.from(self.obj.child(1));
}

pub fn call(function: Value, vm: anytype, args: []const Value) !Value {
    std.debug.assert(function.kind() == .function);
    std.debug.assert(function.arguments().size() == args.len);
    const fun = function.vm_fun();
    const closure = function.captured().obj;
    const all_args = try vm.ally.alloc(Obj, args.len + 1);
    defer vm.ally.free(all_args);
    for (args, 0..) |arg, i| all_args[i] = arg.obj;
    all_args[args.len] = closure;

    var fuel: usize = Vm.max_fuel;
    const result = try vm.call(fun, all_args, &fuel);
    return switch (result) {
        .returned => |obj| .{ .obj = obj },
        .crashed => |e| {
            std.debug.print("\nUncaught crash from a Value.call:\n{f}\n", .{Value.from(e)});
            std.process.exit(1);
        },
        .out_of_fuel => @panic("ran out of fuel (top-level Value.call)"),
        .out_of_memory => return error.OutOfMemory,
    };
}

pub fn format(self: Value, writer: *std.Io.Writer) !void {
    try self.format_indented(writer, 0);
}
pub fn format_singleline(self: Value, writer: *std.Io.Writer) error{WriteFailed}!void {
    const ty = self.obj.child(0);
    switch (self.kind()) {
        .int => try writer.print("{d}", .{self.int()}),
        .float => try writer.print("{d}", .{self.float()}),
        .string => {
            try writer.print("\"", .{});
            for (self.string()) |byte| {
                switch (byte) {
                    '\n' => try writer.print("\\n", .{}),
                    32...127 => try writer.print("{c}", .{byte}),
                    else => try writer.print("\\{x}", .{byte}),
                }
            }
            try writer.print("\"", .{});
        },
        .struct_ => {
            try writer.print("(&", .{});
            for (ty.children()[1..], self.obj.children()[1..]) |field_name, field_| {
                try writer.print(" {s} ", .{Heap.get_symbol(field_name)});
                try (Value{ .obj = field_ }).format_singleline(writer);
            }
            try writer.print(")", .{});
        },
        .enum_ => {
            try writer.print("(| {s}", .{self.variant()});
            const payload_ = self.payload();
            if (payload_.kind() == .struct_ and payload_.obj.size() == 1) {
                // The payload is (&), don't print it.
            } else {
                try writer.print(" ", .{});
                try payload_.format_singleline(writer);
            }
            try writer.print(")", .{});
        },
        .array => {
            try writer.print("([]", .{});
            for (self.items()) |item| {
                try writer.print(" ", .{});
                try item.format_singleline(writer);
            }
            try writer.print(")", .{});
        },
        .function => {
            try writer.print("(\\ (", .{});
            for (self.arguments().children(), 0..) |arg, i| {
                if (i > 0) try writer.print(" ", .{});
                try writer.print("{s}", .{Heap.get_symbol(arg)});
            }
            try writer.print(") ...", .{});
            // try writer.print(") [", .{});
            // try self.get_captured().format_singleline(writer);
            // try writer.print("] ", .{});
            // const ir = self.get_ir();
            // if (ir.size() > 0) {
            //     try Value.from(ir).format_singleline_code(writer);
            // } else {
            //     try writer.print("...", .{});
            // }
            try writer.print(")", .{});
        },
    }
}
pub fn format_singleline_code(self: Value, writer: *std.Io.Writer) !void {
    const variant_ = self.variant();
    const payload_ = self.payload();
    if (std.mem.eql(u8, variant_, "value")) {
        try payload_.format_singleline(writer);
        return;
    }
    if (std.mem.eql(u8, variant_, "name")) {
        try writer.print("{s}", .{payload_.string()});
        return;
    }
    if (std.mem.eql(u8, variant_, "let")) {
        try writer.print("(: {s} ", .{payload_.field("name").string()});
        try payload_.field("definition").format_singleline_code(writer);
        try writer.print(" ", .{});
        try payload_.field("body").format_singleline_code(writer);
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "add") or
        std.mem.eql(u8, variant_, "subtract") or
        std.mem.eql(u8, variant_, "multiply") or
        std.mem.eql(u8, variant_, "divide") or
        std.mem.eql(u8, variant_, "modulo") or
        std.mem.eql(u8, variant_, "compare"))
    {
        try writer.print("(@{s} ", .{variant_});
        try payload_.field("left").format_singleline_code(writer);
        try writer.print(" ", .{});
        try payload_.field("right").format_singleline_code(writer);
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "struct")) {
        try writer.print("(&", .{});
        for (payload_.get_items()) |field_| {
            try writer.print(" {s} ", .{field_.field("name").string()});
            try field_.field("value").format_singleline_code(writer);
        }
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "field")) {
        const of = payload_.field("of");
        const name = payload_.field("name");
        if (name.kind() == .enum_ and std.mem.eql(u8, name.variant_(), "value") and name.payload().kind() == .string) {
            try writer.print("(.{s} ", .{name.payload().string()});
            try of.format_singleline_code(writer);
            try writer.print(")", .{});
        } else {
            try writer.print("(@field ", .{});
            try of.format_singleline_code(writer);
            try writer.print(" ", .{});
            try name.format_singleline_code(writer);
            try writer.print(")", .{});
        }
        return;
    }
    if (std.mem.eql(u8, variant_, "enum")) {
        const actual_variant = payload_.field("variant").string();
        const actual_payload = payload_.field("payload");
        try writer.print("(| {s} ", .{actual_variant});
        try actual_payload.format_singleline_code(writer);
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "switch")) {
        try writer.print("(% ", .{});
        try payload_.field("condition").format_singleline_code(writer);
        for (payload_.field("cases").items()) |case| {
            const case_variant = case.field("variant").string();
            const binding_val = case.field("payload");
            const binding = bind: {
                const bind = binding_val.variant();
                if (std.mem.eql(u8, bind, "none")) break :bind null;
                if (std.mem.eql(u8, bind, "some")) break :bind binding_val.payload().string();
                @panic("invalid IR");
            };
            const body = case.field("body");
            if (binding) |name| {
                try writer.print(" ({s} {s}) ", .{ case_variant, name });
            } else {
                try writer.print(" {s} ", .{case_variant});
            }
            try body.format_singleline_code(writer);
        }
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "make_function")) {
        const args = payload_.field("arguments");
        const captured_ = payload_.field("captured");
        const body = payload_.field("body");
        if (std.mem.eql(u8, args.variant(), "value") and std.mem.eql(u8, body.variant(), "value")) {
            try writer.print("(\\ (", .{});
            for (args.payload().items(), 0..) |arg, i| {
                if (i > 0) try writer.print(" ", .{});
                try writer.print("{s}", .{arg.string()});
            }
            try writer.print(") [", .{});
            try captured_.format_singleline_code(writer);
            try writer.print("] ", .{});
            try body.payload().format_singleline_code(writer);
            try writer.print(")", .{});
        } else {
            try writer.print("(@function ", .{});
            try args.format_singleline_code(writer);
            try writer.print(" ", .{});
            try captured_.format_singleline_code(writer);
            try writer.print(" ", .{});
            try body.format_singleline_code(writer);
            try writer.print(")", .{});
        }
        return;
    }
    if (std.mem.eql(u8, variant_, "call")) {
        try writer.print("(", .{});
        try payload.field("function").format_singleline_code(writer);
        for (payload.field("arguments").items()) |arg| {
            try writer.print(" ", .{});
            try arg.format_singleline_code(writer);
        }
        try writer.print(")", .{});
        return;
    }
    if (std.mem.eql(u8, variant_, "crash")) {
        try writer.print("(@crash ", .{});
        try payload.format_singleline_code(writer);
        try writer.print(")", .{});
        return;
    }
    try writer.print("(@ir ", .{});
    try self.format_singleline(writer);
    try writer.print(")", .{});
}
fn singleline_len(self: Value) !usize {
    var len_tracker = std.Io.Writer.Discarding.init(&[_]u8{});
    try self.format_singleline(&len_tracker.writer);
    return len_tracker.count;
}
const WIDTH_LIMIT = 100;
pub fn format_indented(self: Value, writer: *std.Io.Writer, indentation: usize) !void {
    if (indentation + try self.singleline_len() <= WIDTH_LIMIT) {
        try self.format_singleline(writer);
        return;
    }

    const ty = self.obj.child(0);
    switch (self.kind()) {
        .int => try writer.print("{d}", .{self.int()}),
        .float => try writer.print("{d}", .{self.float()}),
        .string => {
            try writer.print("\"", .{});
            for (self.string()) |byte| {
                switch (byte) {
                    '\n' => try writer.print("\\n", .{}),
                    32...127 => try writer.print("{c}", .{byte}),
                    else => try writer.print("\\{x}", .{byte}),
                }
            }
            try writer.print("\"", .{});
        },
        .struct_ => {
            try writer.print("(&", .{});
            for (ty.children()[1..], self.obj.children()[1..]) |field_name, field_value| {
                const name = Heap.get_symbol(field_name);
                const value = Value{ .obj = field_value };
                try writer.print("\n", .{});
                for (0..indentation + 1) |_| try writer.writeAll(" ");
                try writer.print("{s} ", .{name});
                if (2 * (indentation + 1) + name.len + 1 + try value.singleline_len() <= WIDTH_LIMIT) {
                    try value.format_singleline(writer);
                } else {
                    try writer.print("\n", .{});
                    for (0..indentation + 2) |_| try writer.writeAll(" ");
                    try value.format_indented(writer, indentation + 2);
                }
            }
            try writer.print(")", .{});
        },
        .enum_ => {
            try writer.print("(| {s}", .{self.variant()});
            const payload_ = self.payload();
            if (payload_.kind() == .struct_ and payload_.obj.size() == 1) {
                // The payload is (&), don't print it.
            } else {
                try writer.print("\n", .{});
                for (0..indentation + 1) |_| try writer.writeAll(" ");
                try payload_.format_indented(writer, indentation + 1);
            }
            try writer.print(")", .{});
        },
        .array => {
            try writer.print("([]", .{});
            for (self.items()) |item| {
                try writer.print("\n", .{});
                for (0..indentation + 1) |_| try writer.writeAll(" ");
                try item.format_indented(writer, indentation + 1);
            }
            try writer.print(")", .{});
        },
        .function => {
            try writer.print("(\\ (", .{});
            for (self.arguments().children(), 0..) |arg, i| {
                if (i > 0) try writer.print(" ", .{});
                try writer.print("{s}", .{Heap.get_symbol(arg)});
            }
            try writer.print(")", .{});
            try writer.print(" ...", .{});
            // try writer.print("\n", .{});
            // for (0..indentation + 1) |_| try writer.print(" ", .{});
            // try writer.print("[", .{});
            // try self.captured().format_indented(writer, indentation + 1);
            // try writer.print("]\n", .{});
            // for (0..indentation + 1) |_| try writer.print(" ", .{});
            // const ir_ = self.ir();
            // if (ir_.size() > 0) {
            //     try Value.from(ir_).format_singleline_code(writer);
            // } else {
            //     try writer.print("...", .{});
            // }
            try writer.print(")", .{});
        },
    }
}
