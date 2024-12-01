from manim import *
import numpy as np

class GradientDescentPlotScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set a custom background color (light beige)
        self.camera.background_color = "#ddd7d0"

    def construct(self):
        # -----------------------------
        # 1. Define the functions f(x) and g(x)
        # -----------------------------
        def f(x):
            return 1 + 2 * x

        def g(x, c):
            return 1 + c * x  # General form with coefficient c

        # Initialize coefficient c for g(x)
        c_current = 0  # Start with c=0, so g(x) = 1 + 0x = 1

        # -----------------------------
        # 2. Set up the axes
        # -----------------------------
        # Define the range for x and y axes
        x_min, x_max = -1, 1
        y_min, y_max = -1, 4  # Adjusted to accommodate f(x) = 1 + 2x

        axes = Axes(
            x_range=[x_min, x_max, 0.5],
            y_range=[y_min, y_max, 1],
            axis_config={
                "include_numbers": True,
                "color": BLACK,           # Set axis lines to black
                "stroke_width": 2,
                "font_size": 24,          # Increase font size for axis numbers
                "tip_length": 0.15,       # Length of the arrow tips
            },
            tips=True,  # Include arrow tips
        ).center()  # Center the axes in the scene

        # Add labels to the axes
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set_color(BLACK)

        # -----------------------------
        # 3. Create the initial function graphs
        # -----------------------------
        # Plot f(x) = 1 + 2x in orange
        f_graph = axes.plot(f, color=ORANGE, stroke_width=4, x_range=[x_min, x_max])

        # Plot initial g(x) = 1 in blue (c=0)
        g_graph = axes.plot(lambda x: g(x, c_current), color=BLUE, stroke_width=4, x_range=[x_min, x_max])

        # -----------------------------
        # 4. Define the initial shaded region between f(x) and g(x)
        # -----------------------------
        # Define the color for the shaded region (navy blue)
        navy_blue = "#000080"

        # Get the area between f_graph and g_graph from x_min to x_max
        shaded_region = axes.get_area(
            graph=f_graph,
            x_range=[x_min, x_max],
            bounded_graph=g_graph,
            color=navy_blue,
            opacity=0.5  # 50% transparency
        )

        # -----------------------------
        # 5. Add Residuals Arrow and Label at x = 0.5
        # -----------------------------
        # Define the x-position for the residual
        residual_x = 0.5

        # Calculate y-values for f(x) and g(x) at residual_x
        y_f = f(residual_x)  # 1 + 2*0.5 = 2.0
        y_g = g(residual_x, c_current)  # 1 + 0*0.5 = 1.0

        # Convert coordinates to scene points
        point_f = axes.c2p(residual_x, y_f)
        point_g = axes.c2p(residual_x, y_g)

        # Create a double-headed arrow between f(x) and g(x)
        residual_arrow = DoubleArrow(
            start=point_g,
            end=point_f,
            color=WHITE,
            stroke_width=2,
            buff=0.05,  # Slight buffer to prevent overlapping with graphs
            max_tip_length_to_length_ratio=0.2  # Size of arrow tips
        )

        # Create the label "Residuals"
        residual_label = Text(
            "Residuals",
            color=WHITE,
            font_size=24
        )

        # Create a background rectangle for the "Residuals" text
        residual_label_bg = SurroundingRectangle(
            residual_label,
            color=navy_blue,
            fill_color=navy_blue,
            fill_opacity=1  # Fully opaque background
        )

        # Group the background and text together
        residual_label_group = VGroup(residual_label_bg, residual_label).next_to(
            residual_arrow, RIGHT, buff=0.1
        ).shift(RIGHT * 0.1 + UP * 0.1)  # Slight shift up

        # -----------------------------
        # 6. Create Legend Labels at Bottom Right
        # -----------------------------
        # Define the labels with colored dots and text

        # Label for f(x)
        f_label_dot = Dot(color=ORANGE, radius=0.07)
        f_label_text = Text("f(x) = 1 + 2x", color=ORANGE, font_size=28)
        f_label_group = VGroup(f_label_dot, f_label_text).arrange(RIGHT, buff=0.2)

        # Label for g(x)
        g_label_dot = Dot(color=BLUE, radius=0.07)
        g_label_text = Text("g(x) = 1", color=BLUE, font_size=28)
        g_label_group = VGroup(g_label_dot, g_label_text).arrange(RIGHT, buff=0.2)

        # Combine labels into a single legend
        legend = VGroup(f_label_group, g_label_group).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        legend.to_corner(DOWN + RIGHT, buff=0.5)  # Position at the bottom right corner

        # -----------------------------
        # 7. Animate the initial plotting
        # -----------------------------
        # Add axes and labels instantly
        self.add(axes, axes_labels)

        # Animate drawing f(x) over 2 seconds and its legend label simultaneously
        self.play(
            Create(f_graph),
            FadeIn(f_label_group),
            run_time=2
        )
        self.wait(1)  # Brief pause

        # Animate drawing g(x) over 2 seconds and its legend label simultaneously
        self.play(
            Create(g_graph),
            FadeIn(g_label_group),
            run_time=2
        )
        self.wait(1)  # Brief pause

        # Animate drawing the shaded region over 2 seconds
        self.play(FadeIn(shaded_region), run_time=2)
        self.wait(1)  # Brief pause

        # Animate the residual arrow and label with fade-in
        self.play(
            FadeIn(residual_arrow),
            FadeIn(residual_label_group),  # Updated to include background
            run_time=2
        )
        self.wait(2)  # Keep everything on screen for 2 seconds

        # -----------------------------
        # 8. Add the legend to the scene
        # -----------------------------
        self.play(FadeIn(legend), run_time=1)
        self.wait(1)  # Brief pause

        # -----------------------------
        # 9. Iterative Transformation of g(x) and Shaded Region
        # -----------------------------
        # Define the number of iterations
        num_iterations = 10

        learning_rate = 0.25

        for i in range(1, num_iterations + 1):
            # Update coefficient c
            c_new = c_current + learning_rate * (2 - c_current)  # Since f(x) has a slope of 2

            # Define the new g(x)
            def g_new(x, c=c_new):
                return 1 + c * x

            # Create the new g_graph
            new_g_graph = axes.plot(lambda x: g_new(x), color=BLUE, stroke_width=4, x_range=[x_min, x_max])

            # Define the new shaded region between f_graph and new_g_graph
            new_shaded_region = axes.get_area(
                graph=f_graph,
                x_range=[x_min, x_max],
                bounded_graph=new_g_graph,
                color=navy_blue,
                opacity=0.5  # Maintain same transparency
            )

            # Calculate new residual arrow positions based on new_g(x)
            y_g_new = g_new(residual_x)  # 1 + c_new * 0.5
            point_g_new = axes.c2p(residual_x, y_g_new)

            # Create the new residual arrow between new_g(x) and f(x)
            new_residual_arrow = DoubleArrow(
                start=point_g_new,
                end=point_f,  # f(x) remains the same
                color=WHITE,
                stroke_width=2,
                buff=0.05,
                max_tip_length_to_length_ratio=0.2
            )

            # Create the new residual label text
            new_residual_label = Text(
                "Residuals",
                color=WHITE,
                font_size=24
            )

            # Create a new background rectangle for the updated "Residuals" text
            new_residual_label_bg = SurroundingRectangle(
                new_residual_label,
                color=navy_blue,
                fill_color=navy_blue,
                fill_opacity=1  # Fully opaque background
            )

            # Group the new background and text together
            new_residual_label_group = VGroup(new_residual_label_bg, new_residual_label).next_to(
                new_residual_arrow, RIGHT, buff=0.1
            ).shift(RIGHT * 0.1 + UP * 0.1)  # Slight shift up

            # Create the new residual arrow and label group
            new_residual_group = VGroup(new_residual_arrow, new_residual_label_group)

            # Create the new g(x) legend label text
            new_g_label_text = Text(f"g(x) = 1 + {c_new:.2f}x", color=BLUE, font_size=28)

            # Animate transforming g_graph to new_g_graph, shaded_region to new_shaded_region,
            # residual_arrow to new_residual_arrow, residual_label to new_residual_label_group,
            # and update the legend label for g(x)
            self.play(
                Transform(g_graph, new_g_graph),
                Transform(shaded_region, new_shaded_region),
                Transform(residual_arrow, new_residual_arrow),
                Transform(residual_label_group, new_residual_label_group),
                ReplacementTransform(g_label_text, new_g_label_text),
                run_time=2
            )
            self.wait(1)  # Brief pause

            # Update the legend's g_label_group with the new label
            # Remove the old g_label_group and add the new one
            self.remove(g_label_group)
            g_label_group = VGroup(g_label_dot, new_g_label_text).arrange(RIGHT, buff=0.2)
            legend = VGroup(f_label_group, g_label_group).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            legend.to_corner(DOWN + RIGHT, buff=0.5)  # Reposition to maintain alignment

            # Re-add the updated legend
            self.play(FadeIn(g_label_group), run_time=1)
            self.wait(0.5)  # Brief pause

            # Update c_current for the next iteration
            c_current = c_new

        # -----------------------------
        # 10. Final Pause
        # -----------------------------
        self.wait(2)  # Keep the final scene on screen for 2 seconds
