from manim import *
import numpy as np

class GradientDescent2(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"  # Same background color as previous scene

    def construct(self):
        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return 1 / (1 + x**2)  # True function to be plotted

        # -----------------------------
        # 2. Set random seed for reproducibility
        # -----------------------------
        np.random.seed(42)  # Ensure consistency with previous scene

        # -----------------------------
        # 3. Generate data points (same as previous scene)
        # -----------------------------
        num_points = 500  # Number of data points
        x_min, x_max = -1.8, 1.8  # Range of x values

        # Uniformly distributed x values
        x_values = np.random.uniform(x_min, x_max, num_points)

        # Compute y values with added Gaussian noise
        noise_std = 0.1  # Standard deviation of noise
        y_values = f(x_values) + np.random.normal(0, noise_std, num_points)

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

        self.wait(1)  # Pause at the end of the animation
