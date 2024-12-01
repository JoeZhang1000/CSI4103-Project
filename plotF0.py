from manim import *
import numpy as np

class F0Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"  # Same background color as previous scene

    def construct(self):
        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return  1 + 2*x

        # -----------------------------
        # 2. Set random seed for reproducibility
        # -----------------------------
        np.random.seed(42)  # Ensure consistency with previous scene

        # -----------------------------
        # 3. Generate data points (same as previous scene)
        # -----------------------------
        num_points = 75  # Number of data points
        x_min, x_max = -1.8, 1.8  # Range of x values

        # Uniformly distributed x values
        x_values = np.linspace(x_min, x_max, num_points)

        # Compute y values with added Gaussian noise
        y_values = f(x_values)

        # -----------------------------
        # 4. Create axes (same as previous scene)
        # -----------------------------
        axes = Axes(
            x_range=[x_min, x_max, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"include_numbers": True},
            tips=True,  # Include arrow tips
        ).set_color(BLACK)

        # Add labels to the axes
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set(color=BLACK)

        axes.center()  # Center the axes in the scene

        # -----------------------------
        # 5. Create dots for each data point (same as previous scene)
        # -----------------------------
        # Convert data points to Manim's coordinate system
        points = [
            axes.c2p(x, y) for x, y in zip(x_values, y_values)
        ]

        # Create a Dot for each point with a smaller size for better performance
        dots = VGroup(*[
            Dot(point=point, radius=0.03, color=ORANGE)  # Smaller radius
            for point in points
        ])

        # -----------------------------
        # 6. Add axes and labels to the scene
        # -----------------------------
        self.add(axes, axes_labels)  # Add axes and labels without animation
        self.wait(0.5)  # Brief pause before plotting points

        # -----------------------------
        # 7. Animate plotting the points over 1 second
        # -----------------------------
        # Use the Create animation with run_time=1 second
        self.play(Create(dots), run_time=1)
        self.wait(0.5)  # Brief pause after plotting points

        # -----------------------------
        # 8. Calculate the average of y_values
        # -----------------------------
        avg_y = np.mean(y_values)
        print(f"Average y-value: {avg_y}")  # Optional: For debugging purposes

        # -----------------------------
        # 9. Define the constant average function
        # -----------------------------
        avg_func = lambda x: avg_y

        # -----------------------------
        # 10. Create the average line in red using Line instead of get_graph
        # -----------------------------
        avg_line = Line(
            start=axes.c2p(x_min, avg_y),
            end=axes.c2p(x_max, avg_y),
            color=RED,
            stroke_width=5
        )

        # Optional: Add a label to the average line
        avg_label = MathTex(f"").set_color(RED)  # Added label text
        # Position the label slightly above the left end of the average line
        avg_label.next_to(axes.c2p(x_min, avg_y), UP, buff=0.1)

        # -----------------------------
        # 11. Animate plotting the average line
        # -----------------------------
        self.play(Create(avg_line), Write(avg_label), run_time=1)
        self.wait(1)  # Pause to display the average line

        # -----------------------------
        # 12. Create lines from avg_line to each data point
        # -----------------------------
        # Calculate the starting points on the average line
        start_points = [axes.c2p(x, avg_y) for x in x_values]

        # Create a Line from each start point to the corresponding data point
        connecting_lines = VGroup(*[
            Line(start=start, end=end, color="#030040", stroke_width=1)
            for start, end in zip(start_points, points)
        ])

        # -----------------------------
        # 13. Animate the connecting lines over 5 seconds
        # -----------------------------
        # To optimize performance, we'll create the lines first with zero length and then animate their growth
        connecting_lines.set_stroke(width=1, opacity=0)  # Start invisible

        self.add(connecting_lines)  # Add lines to the scene without displaying them

        # Animate the lines appearing by increasing their opacity and showing them
        self.play(
            connecting_lines.animate.set_opacity(1),
            run_time=5
        )

        # Alternatively, for a "growing" effect from avg_line to data points:
        # self.play(
        #     *[GrowFromPoint(line, line.get_start()) for line in connecting_lines],
        #     run_time=5,
        #     rate_func=linear
        # )

        self.wait(1)  # Pause at the end of the animation

