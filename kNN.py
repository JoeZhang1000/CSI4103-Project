from manim import *
import numpy as np

class KNNResidualsWithZoomScene(MovingCameraScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"

    def construct(self):
        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return 1 + 2 * x

        # -----------------------------
        # 2. Generate data points without noise
        # -----------------------------

        np.random.seed(42)
        num_points = 100
        x_min, x_max = -1.8, 1.8
        x_values = np.linspace(x_min, x_max, num_points) + np.random.normal(0, 0.005, num_points)
        # x_values = np.linspace(x_min, x_max, num_points)
        y_values = f(x_values)

        # -----------------------------
        # 3. Create axes
        # -----------------------------
        axes = Axes(
            x_range=[x_min, x_max, 0.5],
            y_range=[-2, f(x_max) + 1, 1],
            axis_config={"include_numbers": True},
            tips=True,
        ).set_color(BLACK)
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set(color=BLACK)
        axes.center()

        # -----------------------------
        # 4. Plot constant line h at y=1 (colored blue)
        # -----------------------------
        h_value = 1
        h_line = Line(
            start=axes.c2p(x_min, h_value),
            end=axes.c2p(x_max, h_value),
            color=BLUE,
            stroke_width=2
        )

        # -----------------------------
        # 5. Calculate residuals (f(x) - h)
        # -----------------------------
        residuals = y_values - h_value  # Residuals are 2x
        residual_lines = VGroup(*[
            Line(start=axes.c2p(x, h_value), end=axes.c2p(x, y), color=DARK_BLUE, stroke_width=1)
            for x, y in zip(x_values, y_values)
        ])

        # -----------------------------
        # 6. Plot data points
        # -----------------------------
        points = [axes.c2p(x, y) for x, y in zip(x_values, y_values)]
        dots = VGroup(*[
            Dot(point=point, radius=0.03, color=ORANGE)
            for point in points
        ])

        # -----------------------------
        # 7. Add axes and labels to the scene
        # -----------------------------
        self.add(axes, axes_labels)

        # -----------------------------
        # 8. Animate drawing of orange points first
        # -----------------------------
        self.play(Create(dots), run_time=2)
        self.wait(0.5)

        # -----------------------------
        # 9. Animate drawing of blue line
        # -----------------------------
        self.play(Create(h_line), run_time=2)
        self.wait(0.5)

        # -----------------------------
        # 10. Animate drawing of residuals immediately after blue line
        # -----------------------------
        self.play(Create(residual_lines), run_time=2)
        self.wait(1)

        # -----------------------------
        # 11. Define the new target point (perfectly centered between two data points)
        # -----------------------------
        for i in range(len(x_values) - 1):
            if x_values[i] < 0.25 and x_values[i + 1] > 0.25:
                break

        if x_values[i] < 0.2 or x_values[i + 1] > 0.3:
            raise ValueError("Could not find suitable data points between x=0.2 and x=0.3.")

        target_x = (x_values[i] + x_values[i + 1]) / 2
        target_y = f(target_x)
        target_point = axes.c2p(target_x, target_y)
        target_dot = Dot(point=target_point, radius=0.05, color=YELLOW)

        # -----------------------------
        # 12. Add the target point to the scene
        # -----------------------------
        self.play(FadeIn(target_dot), run_time=1)
        self.wait(0.5)

        # -----------------------------
        # 13. Activate Zooming by moving and zooming the camera
        # -----------------------------
        zoom_width = 3  # Adjust to control zoom level
        self.play(
            self.camera.frame.animate.move_to(target_point).set(width=zoom_width),
            run_time=2
        )
        self.wait(6)

        # -----------------------------
        # 14. Find the 4 nearest neighbors based on x-distance
        # -----------------------------
        distances = np.abs(x_values - target_x)
        neighbor_indices = np.argsort(distances)[:4]

        # Highlight the neighbor dots and set their residuals to green
        neighbor_dots = VGroup(*[dots[i] for i in neighbor_indices])
        neighbor_residuals = VGroup(*[residual_lines[i] for i in neighbor_indices])

        self.play(
            *[dot.animate.set_color(GREEN) for dot in neighbor_dots],
            *[residual.animate.set_color(GREEN) for residual in neighbor_residuals],
            run_time=1
        )
        self.wait(0.5)

        # -----------------------------
        # 15. Compute weighted average of residuals (k=4)
        # -----------------------------
        neighbor_residual_values = residuals[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]
        weights = 1 / (neighbor_distances + 1e-10)  # Avoid division by zero
        weights /= weights.sum()
        estimated_residual = np.sum(neighbor_residual_values * weights)

        # -----------------------------
        # 16. Display the estimated residual for the target point (darker blue, less thick)
        # -----------------------------
        estimated_line = Line(
            start=axes.c2p(target_x, h_value),
            end=axes.c2p(target_x, h_value + estimated_residual),
            color="#00008B",  # DarkBlue
            stroke_width=2  # Less thick
        )
        self.play(Create(estimated_line), run_time=1)
        self.wait(1)

        # -----------------------------
        # 17. Show the true line in orange one second after the estimated residual is shown
        #    - Keep the estimated residual on the screen
        #    - Draw the true line underneath all other objects
        # -----------------------------
        true_line = axes.plot(lambda x: f(x), color=ORANGE, stroke_width=2)
        true_line.z_index = -1  # Ensure the true_line is rendered beneath other objects
        self.play(Create(true_line), run_time=1)
        self.wait(2)

        # -----------------------------
        # 18. Add summary text
        # -----------------------------
        summary = Text(
            "kNN Residual Estimation",
            font_size=24,
            color=BLACK
        ).to_edge(UP)
        self.play(Write(summary), run_time=1)
        self.wait(2)

        # -----------------------------
        # 19. Clean up (remove summary only, keep estimated residual and true line)
        # -----------------------------
        self.play(FadeOut(summary))
        self.wait(1)

        # -----------------------------
        # 20. Revert colors of neighbors to original state
        # -----------------------------
        self.play(
            *[dot.animate.set_color(ORANGE) for dot in neighbor_dots],
            *[residual.animate.set_color(DARK_BLUE) for residual in neighbor_residuals],
            run_time=1
        )
        self.wait(1)

        # -----------------------------
        # 21. (Optional) Keep the estimated residual and true line on the screen
        #    - No further actions needed as we did not remove them
        # -----------------------------
