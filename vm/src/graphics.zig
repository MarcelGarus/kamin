const std = @import("std");
const Heap = @import("heap.zig");
const Obj = Heap.Obj;
const Ally = std.mem.Allocator;
const ArrayList = std.ArrayList;
const builtin = @import("builtin");
const gl = @cImport({
    @cInclude("glad/glad.h");
    @cInclude("GLFW/glfw3.h");
});
const nvg = @import("nanovg");
const Value = @import("value.zig");

const Graphics = @This();

pub const Size = struct { width: f64, height: f64 };
pub const Position = struct { x: f64, y: f64 };
pub const Color = struct { r: u8, g: u8, b: u8 };
pub const Path = []const PathSegment;
pub const PathSegment = union(enum) { move_to: Position, line_to: Position };
pub const DrawingInstruction = union(enum) {
    fill_path: struct { path: Path, color: Color },
    translate: struct { by: Position, children: []const DrawingInstruction },
};

pub fn parse_position(value: Value) !Position {
    return .{
        .x = value.field("x").float(),
        .y = value.field("y").float(),
    };
}
pub fn parse_color(value: Value) !Color {
    return .{
        .r = @intCast(value.field("r").int()),
        .g = @intCast(value.field("g").int()),
        .b = @intCast(value.field("b").int()),
    };
}
pub fn parse_segment(value: Value) !PathSegment {
    const variant = value.variant();
    const payload = value.payload();
    if (std.mem.eql(u8, variant, "move-to")) {
        return .{ .move_to = try parse_position(payload) };
    }
    if (std.mem.eql(u8, variant, "line-to")) {
        return .{ .line_to = try parse_position(payload) };
    }
    std.debug.print("segment: {s}\n", .{variant});
    return error.UnknownPathSegment;
}
pub fn parse_path(ally: Ally, value: Value) !Path {
    const items = value.items();
    var path = try ally.alloc(PathSegment, items.len);
    for (items, 0..) |item, i| {
        path[i] = try parse_segment(item);
    }
    return path;
}
pub fn parse_drawing_instructions_rec(
    ally: Ally,
    instructions: *ArrayList(DrawingInstruction),
    value: Value,
) error{ OutOfMemory, UnknownPathSegment }!void {
    const variant = value.variant();
    const payload = value.payload();
    if (std.mem.eql(u8, variant, "nothing")) {
        return;
    }
    if (std.mem.eql(u8, variant, "fill-path")) {
        try instructions.append(ally, .{ .fill_path = .{
            .path = try parse_path(ally, payload.field("path")),
            .color = try parse_color(payload.field("color")),
        } });
        return;
    }
    if (std.mem.eql(u8, variant, "all")) {
        for (payload.items()) |item| {
            try parse_drawing_instructions_rec(ally, instructions, item);
        }
        return;
    }
    if (std.mem.eql(u8, variant, "translate")) {
        const by = try parse_position(payload.field("by"));
        var children = ArrayList(DrawingInstruction).empty;
        try parse_drawing_instructions_rec(ally, &children, payload.field("what"));
        try instructions.append(ally, .{ .translate = .{
            .by = by,
            .children = children.items,
        } });
        return;
    }
    std.debug.print("drawing instruction \"{s}\"\n", .{variant});
    @panic("unknown drawing instruction");
}
pub fn parse_drawing_instructions(ally: Ally, value: Value) ![]const DrawingInstruction {
    var instructions = ArrayList(DrawingInstruction).empty;
    try parse_drawing_instructions_rec(ally, &instructions, value);
    return instructions.items;
}

pub fn dump_drawing_instructions(instructions: []const DrawingInstruction) !void {
    try dump_drawing_instructions_indented(instructions, 0);
}
pub fn dump_drawing_instructions_indented(instructions: []const DrawingInstruction, indentation: usize) !void {
    for (instructions) |instruction| {
        for (0..indentation) |_| std.debug.print(" ", .{});
        switch (instruction) {
            .fill_path => |fill| {
                std.debug.print("fill-path", .{});
                std.debug.print(" #{x:02}{x:02}{x:02}", .{ fill.color.r, fill.color.g, fill.color.b });
                for (fill.path) |segment| {
                    switch (segment) {
                        .move_to => |pos| std.debug.print(" ({d}|{d})", .{ pos.x, pos.y }),
                        .line_to => |pos| std.debug.print("--({d}|{d})", .{ pos.x, pos.y }),
                    }
                }
                std.debug.print("\n", .{});
            },
            .translate => |translate| {
                std.debug.print("translate ({d}|{d})", .{ translate.by.x, translate.by.y });
                std.debug.print("\n", .{});
                try dump_drawing_instructions_indented(translate.children, indentation + 1);
            },
        }
    }
}

ally: Ally,
window: ?*gl.GLFWwindow = null,
scale: f32,
vg: nvg,
event_queue: ArrayList(Event),

pub const Event = union(enum) {
    entered_char: struct { codepoint: usize },
    char_entered: CharEntered,
    key_pressed: KeyPressed,

};

fn glfwErrorCallback(error_code: c_int, description: [*c]const u8) callconv(.c) void {
    std.debug.print("GLFW Error [{d}]: {s}\n", .{ error_code, description });
}
pub fn init(ally: Ally) !*Graphics {
    // Initialize libraries.
    _ = gl.glfwSetErrorCallback(glfwErrorCallback);
    if (gl.glfwInit() == gl.GLFW_FALSE) return error.GLFWInitFailed;
    gl.glfwWindowHint(gl.GLFW_CONTEXT_VERSION_MAJOR, 2);
    gl.glfwWindowHint(gl.GLFW_CONTEXT_VERSION_MINOR, 0);
    gl.glfwWindowHint(gl.GLFW_SAMPLES, 4);

    // Get graphics scale.
    const monitor = gl.glfwGetPrimaryMonitor();
    var scale: f32 = 1;
    if (!builtin.target.os.tag.isDarwin())
        gl.glfwGetMonitorContentScale(monitor, &scale, null);
    scale *= 0.7;

    // Create a window.
    const window = gl.glfwCreateWindow(
        @as(i32, @intFromFloat(scale * 500)),
        @as(i32, @intFromFloat(scale * 300)),
        "Orchard",
        null,
        null,
    );
    if (window == null) return error.WindowCreationFailed;
    _ = gl.glfwSetCharCallback(window, charCallback);
    _ = gl.glfwSetKeyCallback(window, keyCallback);
    gl.glfwMakeContextCurrent(window);
    if (gl.gladLoadGL() == 0) return error.GLADInitFailed;
    gl.glfwSwapInterval(1);
    gl.glfwSetTime(0);
    const vg = try nvg.gl.init(ally, .{ .stencil_strokes = true, .debug = true });

    // Bundle everything up in a graphics object and store that in the window.
    const graphics = try ally.create(Graphics);
    graphics.* = .{
        .ally = ally,
        .window = window,
        .scale = scale,
        .vg = vg,
        .event_queue = ArrayList(Event).empty,
    };
    gl.glfwSetWindowUserPointer(window, graphics);
    return graphics;
}

fn charCallback(window: ?*gl.GLFWwindow, codepoint: c_uint) callconv(.c) void {
    const graphics: *Graphics = @ptrCast(@alignCast(gl.glfwGetWindowUserPointer(window)));
    graphics.event_queue.append(graphics.ally, .{
        .char_entered = .{ .codepoint = codepoint },
    }) catch @panic("couldn't append event");
}

fn keyCallback(window: ?*gl.GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.c) void {
    _ = scancode;
    const graphics: *Graphics = @ptrCast(@alignCast(gl.glfwGetWindowUserPointer(window)));
    if (action == gl.GLFW_PRESS or action == gl.GLFW_REPEAT) {
        graphics.event_queue.append(graphics.ally, .{
            .pressed_key = .{
                .keycode = @intCast(key),
                .control = mods & gl.GLFW_MOD_CONTROL != 0,
                .alt = mods & gl.GLFW_MOD_ALT != 0,
                .shift = mods & gl.GLFW_MOD_SHIFT != 0,
            },
        }) catch @panic("couldn't append event");
    }
}

pub fn deinit(self: *Graphics) void {
    gl.glfwDestroyWindow(self.window);
    self.vg.deinit();
    gl.glfwTerminate();
}

pub fn should_close(self: Graphics) bool {
    return gl.glfwWindowShouldClose(self.window) == gl.GLFW_TRUE;
}

pub fn get_mouse_pos(self: Graphics) Position {
    var double_mx: f64 = undefined;
    var double_my: f64 = undefined;
    gl.glfwGetCursorPos(self.window, &double_mx, &double_my);
    const scale: f64 = @floatCast(self.scale);
    return .{ .x = double_mx / scale, .y = double_my / scale };
}

pub fn get_size(self: Graphics) Size {
    var int_width: i32 = undefined;
    var int_height: i32 = undefined;
    gl.glfwGetWindowSize(self.window, &int_width, &int_height);
    const scale: f64 = @floatCast(self.scale);
    return .{
        .width = @as(f64, @floatFromInt(int_width)) / scale,
        .height = @as(f64, @floatFromInt(int_height)) / scale,
    };
}

pub fn get_time(self: Graphics) f32 {
    _ = self;
    const double_t = gl.glfwGetTime();
    const t = @as(f32, @floatCast(double_t));
    return t;
}

// Wakes up the event loop.
pub fn wake() void {
    gl.glfwPostEmptyEvent();
}

pub fn poll_events(self: Graphics) void {
    _ = self;
    // We need to give GLFW the chance to process events internally. There are
    // several ways to do that:
    // - PollEvents() processes all queued-up events, then returns. Good for
    //   games, where stuff happens all the time and you always need new frames.
    // - WaitEvents() blocks and puts the thread asleep until at least one event
    //   happens. Great for editors, etc. because of energy-efficiency.
    // - WaitEventsTimeout(0.7) blocks with a maximum duration in seconds. Good
    //   for non-games with occasional updates such as a clock.
    // See https://www.glfw.org/docs/latest/input_guide.html
    gl.glfwWaitEvents();
}

pub fn render(self: Graphics, size: Size, instructions: []const DrawingInstruction) !void {
    // Calculate pixel ratio for hi-dpi devices.
    var fb_width: i32 = undefined;
    var fb_height: i32 = undefined;
    gl.glfwGetFramebufferSize(self.window, &fb_width, &fb_height);
    const pxRatio = @as(f32, @floatFromInt(fb_width)) / @as(f32, @floatCast(size.width));

    // Update and render
    gl.glViewport(0, 0, fb_width, fb_height);
    gl.glClearColor(1, 1, 1, 1);
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT);

    self.vg.beginFrame(@floatCast(size.width), @floatCast(size.height), pxRatio);

    try render_all(self.vg, instructions);

    self.vg.endFrame();
    gl.glfwSwapBuffers(self.window);
}

fn render_all(vg: nvg, instructions: []const DrawingInstruction) !void {
    for (instructions) |instruction| try render_single(vg, instruction);
}
fn render_single(vg: nvg, instruction: DrawingInstruction) error{OutOfMemory}!void {
    switch (instruction) {
        // .draw_rectangle => |args| {
        //     const x: f32 = @floatFromInt(args.pos.x);
        //     const y: f32 = @floatFromInt(args.pos.y);
        //     const width: f32 = @floatFromInt(args.size.width);
        //     const height: f32 = @floatFromInt(args.size.height);
        //     vg.beginPath();
        //     vg.moveTo(x, y);
        //     vg.lineTo(x + width, y);
        //     vg.lineTo(x + width, y + height);
        //     vg.lineTo(x, y + height);
        //     vg.fillColor(nvg.rgb(args.color.r, args.color.g, args.color.b));
        //     vg.fill();
        // },
        .fill_path => |args| {
            vg.beginPath();
            for (args.path) |segment| {
                switch (segment) {
                    .move_to => |pos| vg.moveTo(@floatCast(pos.x), @floatCast(pos.y)),
                    .line_to => |pos| vg.lineTo(@floatCast(pos.x), @floatCast(pos.y)),
                }
            }
            vg.fillColor(nvg.rgb(args.color.r, args.color.g, args.color.b));
            vg.fill();
        },
        .translate => |t| {
            vg.save();
            vg.translate(@floatCast(t.by.x), @floatCast(t.by.y));
            try render_all(vg, t.children);
            vg.restore();
        },
    }
}
    i = 1;
    while (i < 6) : (i += 1)
        vg.bezierTo(sx[i - 1] + dx * 0.5, sy[i - 1], sx[i] - dx * 0.5, sy[i], sx[i], sy[i]);
    vg.strokeColor(nvg.rgba(0, 160, 192, 255));
    vg.strokeWidth(3.0);
    vg.stroke();

    // Graph sample pos
    i = 0;
    while (i < 6) : (i += 1) {
        bg = vg.radialGradient(sx[i], sy[i] + 2, 3.0, 8.0, nvg.rgba(0, 0, 0, 32), nvg.rgba(0, 0, 0, 0));
        vg.beginPath();
        vg.rect(sx[i] - 10, sy[i] - 10 + 2, 20, 20);
        vg.fillPaint(bg);
        vg.fill();
    }

    vg.beginPath();
    i = 0;
    while (i < 6) : (i += 1)
        vg.circle(sx[i], sy[i], 4.0);
    vg.fillColor(nvg.rgba(0, 160, 192, 255));
    vg.fill();
    vg.beginPath();
    i = 0;
    while (i < 6) : (i += 1)
        vg.circle(sx[i], sy[i], 2.0);
    vg.fillColor(nvg.rgba(220, 220, 220, 255));
    vg.fill();

    vg.strokeWidth(1.0);
}
