// The Heap
//
// This is a heap for immutable objects. When you allocate an object, you have to provide data to initialize
// the memory with. Afterwards, this memory can no longer be changed.
// While the heap has no understanding of what kinds of objects it stores (structs, enums, lambdas, etc.),
// it does know how objects are connected. It knows what parts of the memory are pointers to other
// objects and what parts are literal values. Using this information, the heap can deduplicate objects and
// do garbage collection.
//
// The heap supports two kinds of objects:
// - Leaf objects contain literal words.
// - Inner objects contain pointers to other objects.

const std = @import("std");
const ArrayList = std.ArrayList;
const Map = std.array_hash_map.Auto;
const Ally = std.mem.Allocator;
const Writer = std.Io.Writer;
const writeSliceEndian = Writer.writeSliceEndian;
const ObjMap = @import("obj_map.zig").ObjMap;

const Heap = @This();

comptime {
    if (@sizeOf(usize) != @sizeOf(u64)) @compileError("The heap only works on 64-bit systems.");
}

// The heap is word-based.
pub const Word = usize;

// In memory, every object has the following layout:
//
// [header][word][word]...

pub const Header = packed struct {
    num_words: u48, // 6 bytes, max 281474976710655 words, ca. 2 PiB
    padding1: u8 = 0,
    padding2: u6 = 0,
    marked: u1 = 0, // marking for mark-sweep garbage collection
    is_inner: u1, // is this an inner object (1) or a leaf object (0)?
};
comptime {
    if (@sizeOf(Header) != @sizeOf(Word)) @compileError("bad header layout");
}

pub const Obj = packed struct {
    address: Word,

    pub fn header(self: Obj) *Header {
        return @ptrFromInt(self.address);
    }

    pub fn is_inner(self: Obj) bool {
        return self.header().is_inner == 1;
    }
    pub fn is_leaf(self: Obj) bool {
        return !self.is_inner();
    }

    pub fn size(self: Obj) usize {
        return @intCast(self.header().num_words);
    }

    pub fn words(self: Obj) []const Word {
        if (!self.is_leaf()) unreachable;
        return @as([*]Word, @ptrFromInt(self.address))[1..][0..self.header().num_words];
    }

    pub fn children(self: Obj) []const Obj {
        if (!self.is_inner()) unreachable;
        return @as([*]Obj, @ptrFromInt(self.address))[1..][0..self.header().num_words];
    }

    pub fn word(self: Obj, index: usize) Word {
        return self.words()[index];
    }
    pub fn child(self: Obj, index: usize) Obj {
        return self.children()[index];
    }

    pub fn format_string(chars: []const u8, colored: bool, writer: *std.Io.Writer) error{WriteFailed}!void {
        if (colored) try writer.print("\x1b[48;5;222m", .{});
        for (chars) |char| {
            if (char == 0) break;
            try writer.print("{c}", .{char});
        }
        if (colored) try writer.print("\x1b[0m", .{});
    }
    pub fn format_number(word_: Word, colored: bool, writer: *std.Io.Writer) error{WriteFailed}!void {
        if (colored) try writer.print("\x1b[48;5;153m", .{});
        try writer.print("{d}", .{word_});
        if (colored) try writer.print("\x1b[0m", .{});
    }
    pub fn format_leaf(words_: []const Word, colored: bool, writer: *std.Io.Writer) error{WriteFailed}!void {
        if (words_.len == 1) {
            var only_ascii = true;
            for (@as([]const u8, @ptrCast(words_))) |char| {
                if (char == 0) break;
                if (!(char >= 32 and char <= 126)) only_ascii = false;
            }
            if (words_[0] == 0) only_ascii = false;
            if (only_ascii) {
                try format_string(@as([]const u8, @ptrCast(words_)), colored, writer);
            } else {
                try format_number(words_[0], colored, writer);
            }
        } else {
            try format_string(@as([]const u8, @ptrCast(words_)), colored, writer);
        }
    }
    pub fn format_singleline(self: Obj, colored: bool, writer: *std.Io.Writer) error{WriteFailed}!void {
        if (self.is_leaf()) {
            try format_leaf(self.words(), colored, writer);
        } else {
            try writer.print("(", .{});
            for (self.children(), 0..) |c, i| {
                if (i > 0) {
                    try writer.print(" ", .{});
                }
                try c.format_singleline(colored, writer);
            }
            try writer.print(")", .{});
        }
    }
    fn singleline_len(self: Obj) !usize {
        var len_tracker = std.Io.Writer.Discarding.init(&[_]u8{});
        try self.format_singleline(false, &len_tracker.writer);
        return len_tracker.count;
    }
    const WIDTH_LIMIT = 120;
    pub fn format_indented(self: Obj, writer: *std.Io.Writer, indentation: usize) !void {
        if (indentation + try self.singleline_len() <= WIDTH_LIMIT) {
            try self.format_singleline(true, writer);
            return;
        }

        if (self.is_leaf()) {
            try format_leaf(self.words(), true, writer);
        } else {
            try writer.print("(", .{});
            for (self.children(), 0..) |c, i| {
                if (i > 0) {
                    try writer.print("\n", .{});
                    for (0..indentation + 1) |_| try writer.print(" ", .{});
                }
                try c.format_indented(writer, indentation + 1);
            }
            try writer.print(")", .{});
        }
    }
    pub fn format(obj: Obj, writer: *Writer) !void {
        try obj.format_indented(writer, 0);
    }
};

pub fn is_same(a: Obj, b: Obj) bool {
    if (a.is_inner() != b.is_inner()) return false;
    if (a.size() != b.size()) return false;
    if (a.is_inner()) {
        for (a.children(), b.children()) |ac, bc| if (!is_same(ac, bc)) return false;
    } else {
        for (a.words(), b.words()) |aw, bw| if (aw != bw) return false;
    }
    return true;
}

memory: []Word,
used: usize,

pub fn init(ally: Ally, capacity: usize) !Heap {
    const memory = try ally.alloc(Word, capacity);
    return .{ .memory = memory, .used = 0 };
}

pub fn is_full(heap: Heap) bool {
    return heap.used == heap.memory.len;
}
fn emit(heap: *Heap, word: Word) !void {
    if (heap.is_full()) return error.OutOfMemory;
    heap.memory[heap.used] = word;
    heap.used += 1;
}

pub fn build_leaf(heap: *Heap) !LeafObjBuilder {
    const address = heap.used;
    try heap.emit(@bitCast(Header{ .num_words = 0, .is_inner = 0 }));
    return .{ .heap = heap, .start = address, .num_words = 0 };
}
pub const LeafObjBuilder = struct {
    heap: *Heap,
    start: Word,
    num_words: usize,

    pub fn emit(builder: *LeafObjBuilder, word: Word) !void {
        try builder.heap.emit(word);
        builder.num_words += 1;
    }
    pub fn finish(builder: *LeafObjBuilder) Obj {
        const header: *Header = @ptrCast(&builder.heap.memory[builder.start]);
        header.num_words = @intCast(builder.num_words);
        return .{ .address = @intFromPtr(header) };
    }
};
pub fn new_leaf(heap: *Heap, words: []const Word) !Obj {
    var builder = try heap.build_leaf();
    for (words) |word| try builder.emit(word);
    return builder.finish();
}

pub fn build_inner(heap: *Heap) !InnerObjBuilder {
    const address = heap.used;
    try heap.emit(@bitCast(Header{ .num_words = 0, .is_inner = 1 }));
    return .{ .heap = heap, .start = address, .num_words = 0 };
}
pub const InnerObjBuilder = struct {
    heap: *Heap,
    start: usize,
    num_words: usize,

    pub fn emit(builder: *InnerObjBuilder, obj: Obj) !void {
        try builder.heap.emit(obj.address);
        builder.num_words += 1;
    }
    pub fn finish(builder: *InnerObjBuilder) Obj {
        const header: *Header = @ptrCast(&builder.heap.memory[builder.start]);
        header.num_words = @intCast(builder.num_words);
        return .{ .address = @intFromPtr(header) };
    }
};
pub fn new_inner(heap: *Heap, objs: []const Obj) !Obj {
    var builder = try heap.build_inner();
    for (objs) |obj| try builder.emit(obj);
    return builder.finish();
}

pub const Checkpoint = struct { address: Word };

pub fn start(heap: Heap) Checkpoint {
    return .{ .address = @intFromPtr(heap.memory.ptr) };
}
pub fn checkpoint(heap: Heap) Checkpoint {
    return .{ .address = @intFromPtr(heap.memory.ptr) + 8 * heap.used };
}
pub fn restore(heap: *Heap, checkpoint_: Checkpoint) void {
    heap.used = (checkpoint_.address - @intFromPtr(heap.memory.ptr)) / 8;
}

pub fn deduplicate(heap: *Heap, ally: Ally, from: Checkpoint) !Map(Obj, Obj) {
    var read = from.address;
    var write = from.address;
    var map = Map(Obj, Obj).init(ally);
    const heap_end = heap.checkpoint().address;
    while (read < heap_end) {
        // Adjust the pointers using the mapping so far.
        const header: Header = @as(*Header, @ptrFromInt(read)).*;
        if (header.is_inner == 1) {
            for (0..header.num_words) |i| {
                const ptr: *Obj = @ptrFromInt(read + 8 * (1 + i));
                if (map.get(ptr.*)) |to| ptr.* = to;
            }
        }
        const size = 1 + header.num_words;
        // Compare to existing objects.
        var it = map.iterator();
        while (it.next()) |mapping| {
            const candidate = mapping.value_ptr.*;
            const candidate_header: Header = @as(*Header, @ptrFromInt(candidate.address)).*;
            if (header.num_words != candidate_header.num_words) continue;
            if (std.mem.eql(
                Word,
                @as([*]Word, @ptrFromInt(read))[0..size],
                @as([*]Word, @ptrFromInt(candidate.address))[0..size],
            )) {
                try map.put(@bitCast(read), candidate);
                break;
            }
        } else {
            // Not equal to an existing one. Copy it to the beginning of the compressed heap.
            if (read != write) {
                for (0..size) |i|
                    @as(*Word, @ptrFromInt(write + 8 * i)).* = @as(*Word, @ptrFromInt(read + 8 * i)).*;
            }
            try map.put(@bitCast(read), @bitCast(write));
            write += 8 * size;
        }
        read += 8 * size;
    }
    heap.used = (write - @intFromPtr(heap.memory.ptr)) / 8;
    return map;
}

pub const GcResult = struct { keep: Obj, stable_until: Word };

pub fn garbage_collect(heap: *Heap, ally: Ally, boundary: Checkpoint, keep: Obj) !GcResult {
    mark(keep, boundary);
    return try heap.sweep(ally, keep, boundary);
}
fn mark(obj: Obj, boundary: Checkpoint) void {
    if (obj.address < boundary.address) return;
    const header: *Header = @ptrFromInt(obj.address);
    if (header.marked == 1) return;
    header.marked = 1;
    if (header.is_inner == 1) for (obj.children()) |child| mark(child, boundary);
}
fn sweep(heap: *Heap, ally: Ally, keep: Obj, boundary: Checkpoint) !GcResult {
    var read = boundary.address;
    var write = boundary.address;
    var mapping = Map(Word, Word).empty;
    defer mapping.deinit(ally);
    // The destination of the first object that actually moves. Live objects that
    // are compacted without moving (read == write, i.e. before the first gap)
    // keep their address; everything from here up is relocated.
    const heap_end = heap.checkpoint().address;
    while (read < heap_end) {
        // std.debug.print("read: {x} of {x}\n", .{read, heap_end});
        const header: *Header = @ptrFromInt(read);
        const size = 1 + header.num_words;
        if (header.marked == 0) {
            read += 8 * size; // We don't need this object. Just overwrite it.
        } else {
            header.marked = 0;
            if (read != write) {
                if (first_moved == null) first_moved = write;
                if (header.is_inner == 1) {
                    for (0..header.num_words) |i| {
                        const ptr: *Word = @ptrFromInt(read + 8 * (1 + i));
                        if (mapping.get(ptr.*)) |to| ptr.* = to;
                    }
                }
                for (0..size) |i| @as(*Word, @ptrFromInt(write + 8 * i)).* = @as(*Word, @ptrFromInt(read + 8 * i)).*;
                try mapping.put(ally, read, write);
            }
            read += 8 * size;
            write += 8 * size;
        }
    }
    heap.used = (write - @intFromPtr(heap.memory.ptr)) / 8;
    return .{
        .stable_until = first_moved orelse write,
        .keep = if (mapping.get(keep.address)) |mapped| .{ .address = mapped } else keep,
    };
}

pub fn file_out(obj: Obj, ally: Ally, writer: *std.Io.Writer) !void {
    var name_generator = NameGenerator{};
    var mapping = ObjMap([]const u8).empty;
    try file_out_rec(obj, ally, writer, &name_generator, &mapping);
    try writer.print("export {s}\n", .{mapping.get(obj).?});
    try writer.flush();
}
const NameGenerator = struct {
    count: usize = 0,
    fn generate(gen: *NameGenerator, ally: Ally) ![]const u8 {
        var buffer: [10]u8 = undefined;
        var i: i64 = @intCast(gen.count);
        var j = buffer.len;
        while (i >= 0) {
            j -= 1;
            buffer[j] = @intCast('a' + @mod(i, 26));
            i = @divTrunc(i, 26) - 1;
        }
        const name = buffer[j..];
        const copy = try ally.alloc(u8, name.len);
        std.mem.copyForwards(u8, copy, name);
        gen.count += 1;
        return copy;
    }
};
fn file_out_rec(obj: Obj, ally: Ally, writer: *std.Io.Writer, name_generator: *NameGenerator, exported: *ObjMap([]const u8)) !void {
    if (exported.get(obj)) |_| return;
    if (obj.is_leaf()) {
        const name = try name_generator.generate(ally);
        try exported.put(ally, obj, name);
        try writer.print("{s} \"", .{name});
        var bytes: []const u8 = @ptrCast(obj.words());
        for (0..7) |_| {
            if (bytes.len > 0 and bytes[bytes.len - 1] == 0) bytes.len -= 1;
        }
        for (bytes) |byte| {
            switch (byte) {
                '\\' => try writer.print("\\\\", .{}),
                '"' => try writer.print("\\\"", .{}),
                '\n' => try writer.print("\\n", .{}),
                ' '...'!', '#'...'[', ']'...'~' => try writer.print("{c}", .{byte}),
                else => try writer.print("\\{x:02}", .{byte}),
            }
        }
        try writer.print("\"\n", .{});
    } else {
        for (obj.children()) |child| {
            try file_out_rec(child, ally, writer, name_generator, exported);
        }
        const name = try name_generator.generate(ally);
        try exported.put(ally, obj, name);
        try writer.print("{s} (", .{name});
        for (obj.children(), 0..) |child, i| {
            if (i > 0) try writer.print(" ", .{});
            try writer.print("{s}", .{exported.get(child).?});
        }
        try writer.print(")\n", .{});
    }
}

pub fn copy_to_other_heap(from: *Heap, ally: Ally, to: *Heap, object: Obj) !Obj {
    mark(object, .{ .address = @intFromPtr(from.memory.ptr) });
    var mapping = ObjMap(Obj).empty;
    defer mapping.deinit(ally);
    var cursor = Obj{ .address = @intFromPtr(from.memory.ptr) };
    const end = @intFromPtr(from.memory.ptr) + 8 * from.used;
    while (cursor.address < end) {
        const header_size = cursor.size();
        if (cursor.header().marked == 1) {
            cursor.header().marked = 0;
            var mapped: Obj = undefined;
            if (cursor.is_inner()) {
                var b = try to.build_inner();
                for (cursor.children()) |child| {
                    try b.emit(mapping.get(child) orelse unreachable);
                }
                mapped = b.finish();
            } else {
                mapped = try to.new_leaf(cursor.words());
            }
            try mapping.put(ally, cursor, mapped);
        }
        cursor.address += 8 * (1 + header_size);
    }
    return mapping.get(object) orelse unreachable;
}

pub fn new_symbol(heap: *Heap, value: []const u8) !Obj {
    var b = try heap.build_leaf();
    const num_words = (value.len + 7) / 8;
    for (0..num_words) |i| {
        var w: Word = 0;
        for (0..@min(value.len - i * 8, 8)) |j|
            w |= @as(Word, value[i * 8 + j]) << @intCast(j * 8);
        try b.emit(w);
    }
    return b.finish();
}
pub fn get_symbol(obj: Obj) []const u8 {
    var chars: []const u8 = @ptrCast(obj.words());
    while (chars.len > 0 and chars[chars.len - 1] == 0) chars.len -= 1;
    return chars;
}

pub fn dump_raw(heap: Heap) void {
    for (0.., heap.memory[0..heap.used]) |i, w| {
        std.debug.print("{x:3} |", .{i});
        const bytes: []const u8 = @ptrCast(&w);
        for (bytes) |byte| std.debug.print(" {x:02}", .{byte});
        std.debug.print(" | {}\n", .{w});
    }
}
pub fn dump(heap: Heap) void {
    var i: usize = 0;
    while (i < heap.used) {
        std.debug.print("{x:3} | ", .{i});
        const header: Header = @bitCast(heap.memory[i]);
        std.debug.print("({} words)", .{header.num_words});
        for (0..header.num_words) |j|
            std.debug.print(" {x}", .{heap.memory[i + 1 + j]});
        std.debug.print("\n", .{});
        i += 1 + header.num_words;
    }
}
pub fn dump_stats(heap: Heap) void {
    const num_words = heap.used;
    std.debug.print("{} words", .{num_words});

    std.debug.print(", ", .{});
    const num_bytes = num_words * 8;
    for ([_][]const u8{ "B", "KiB", "MiB", "GiB", "TiB" }, 0..) |unit, index| {
        const amount = num_bytes / std.math.pow(usize, 1024, index);
        if (amount < 1024) {
            std.debug.print("{} {s}", .{ amount, unit });
            break;
        }
    } else std.debug.print("a lot of memory", .{});

    var num_objects: usize = 0;
    var i: usize = 0;
    while (i < num_words) {
        num_objects += 1;
        const header: Header = @bitCast(heap.memory[i]);
        i += 1 + header.num_words;
    }
    std.debug.print(", {} objects\n", .{num_objects});
}
