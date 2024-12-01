from manim import *
import numpy as np


class GradientDescentAndBoostingPlotScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set a custom background color (light beige)
        self.camera.background_color = "#ddd7d0"

    def construct(self):
        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return 1 + 2 * x

        # -----------------------------
        # 2. Set up the axes
        # -----------------------------
        # Define the range for x and y axes
        x_min, x_max = -1.5, 1.5

        axes = Axes(
            x_range=[x_min, x_max, 0.5],
            y_range=[-2, 4, 1],  # Changed step size to 1 for y-axis
            axis_config={
                "include_numbers": True,
                "color": BLACK,          # Set axis color to BLACK
                "stroke_width": 2,
                "font_size": 35,         # Base font size for axis numbers
            },
            tips=True,  # Include arrow tips
        ).set_color(BLACK)  # Ensure the entire axes object is colored BLACK

        # -----------------------------
        # 2a. Update Axis Labels to Use MathTex
        # -----------------------------
        axes_labels = axes.get_axis_labels(
            x_label=MathTex("x", font_size=36),  # Use MathTex for LaTeX math formatting
            y_label=MathTex("y", font_size=36)   # Use MathTex for LaTeX math formatting
        ).set_color(BLACK)  # Set labels color to BLACK

        axes.center()  # Center the axes in the scene

        # -----------------------------
        # 3. Create the true function graph f(x)
        # -----------------------------
        # Plot f(x) within the constrained y-range
        f_graph = axes.plot(
            f,
            x_range=[-1.5, 1.5],  # Adjusted x-range to keep y-values within -2 to 4
            color=ORANGE,
            stroke_width=4,
            name="f(x)"
        )

        # -----------------------------
        # 4. Create Legends for f(x), GD, and GB
        # -----------------------------
        # Adjust legend positions and increase font sizes

        # Legend for f(x)
        f_label = MathTex(r"f(x)", color=ORANGE, font_size=36)  # Increased font size
        f_label_group = VGroup(Dot(color=ORANGE, radius=0.07), f_label).arrange(RIGHT, buff=0.2)

        # Legend for GD
        gd_fixed_label = MathTex(r"g_1(x) = ", color="#000080", font_size=36)  # Changed to Navy Blue
        gd_variable_label = MathTex("1.00 + 0.00x", color="#000080", font_size=36)  # Changed to Navy Blue
        gd_label_group = VGroup(Dot(color="#000080", radius=0.07), gd_fixed_label, gd_variable_label).arrange(RIGHT, buff=0.1)

        # Legend for GB
        gb_fixed_label = MathTex(r"g_2(x) = ", color=BLUE, font_size=36)  # Changed to Blue
        gb_variable_label = MathTex("1.00", color=BLUE, font_size=36)  # Start with "1.00" only, Changed to Blue
        gb_label_group = VGroup(Dot(color=BLUE, radius=0.07), gb_fixed_label, gb_variable_label).arrange(RIGHT, buff=0.1)

        # Combine all legends
        legend = VGroup(f_label_group, gd_label_group, gb_label_group).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        legend.to_corner(DOWN + RIGHT, buff=0.5)  # Move legend closer to the bottom-right corner

        # -----------------------------
        # 5. Initialize GD and GB Models
        # -----------------------------
        # Gradient Descent (GD) parameters
        beta_gd = np.array([1.0, 0.0])  # Initialize beta0=1, beta1=0
        learning_rate_gd = 0.25

        # Gradient Boosting (GB) parameters
        beta_gb = np.array([1.0, 0.0])  # Initialize beta0=1, beta1=0
        learning_rate_gb = 0.25

        # Define GD and GB functions
        def g_gd(x, beta=beta_gd):
            return beta[0] + beta[1] * x

        def g_gb(x, beta=beta_gb):
            return beta[0] + beta[1] * x if beta[1] != 0 else beta[0]  # Start as constant

        # Create initial GD and GB model graphs within the constrained y-range
        g_gd_graph = axes.plot(
            lambda x: g_gd(x),
            x_range=[x_min, x_max],
            color="#000080",  # Changed to Navy Blue
            stroke_width=4,
            name="GD Model"
        )

        g_gb_graph = axes.plot(
            lambda x: g_gb(x),
            x_range=[x_min, x_max],
            color=BLUE,  # Changed to Blue
            stroke_width=4,
            name="GB Model"
        )

        # -----------------------------
        # 6. Add All Objects to the Scene Instantly
        # -----------------------------
        self.add(
            axes,
            axes_labels,
            f_graph,
            g_gd_graph,
            g_gb_graph,
            legend
        )

        # Create iteration number text in the top left corner with larger font size
        iteration_text = MathTex(r"\text{Iteration: } 0", font_size=50).set_color(BLACK).to_corner(UL, buff=0.5)
        self.add(iteration_text)

        self.wait(0.5)  # Brief pause

        # -----------------------------
        # 7. Iterative Updates for GD and GB
        # -----------------------------
        num_iterations = 10

        for i in range(1, num_iterations + 1):
            # -----------------------------
            # A. Gradient Descent (GD) Update
            # -----------------------------
            # Simulate GD step: Move beta_gd towards the true slope
            beta_gd_new = np.array([
                beta_gd[0] + learning_rate_gd * (1 - beta_gd[0]),  # Update beta0 towards 1
                beta_gd[1] + learning_rate_gd * (2 - beta_gd[1])   # Update beta1 towards 2
            ])

            # Create new GD graph
            new_g_gd_graph = axes.plot(
                lambda x: g_gd(x, beta=beta_gd_new),
                x_range=[x_min, x_max],
                color="#000080",  # Changed to Navy Blue
                stroke_width=4,
                name=f"GD Model {i}"
            )

            # Animate GD transformation first
            self.play(
                Transform(g_gd_graph, new_g_gd_graph),
                run_time=1.0  # Adjust run_time as needed
            )
            self.wait(0.1)  # Brief pause

            # Update GD variable label
            gd_new_variable_label = MathTex(
                f"{beta_gd_new[0]:.2f} + {beta_gd_new[1]:.2f}x",
                color="#000080",  # Changed to Navy Blue
                font_size=36
            )
            gd_new_variable_label.next_to(gd_fixed_label, RIGHT, buff=0.1)  # Position next to fixed label

            # Animate GD label transformation
            self.play(
                ReplacementTransform(gd_variable_label, gd_new_variable_label),
                run_time=0.5  # Fast update
            )

            # Update the GD variable label reference
            gd_variable_label = gd_new_variable_label

            # -----------------------------
            # B. Gradient Boosting (GB) Update
            # -----------------------------
            # Simulate GB step: Update beta_gb towards the true slope
            beta_gb_new = np.array([
                beta_gb[0] + learning_rate_gb * (1 - beta_gb[0]),  # Update beta0 towards 1
                beta_gb[1] + learning_rate_gb * (2 - beta_gb[1])   # Update beta1 towards 2
            ])

            # Create new GB graph
            new_g_gb_graph = axes.plot(
                lambda x: g_gb(x, beta=beta_gb_new),
                x_range=[x_min, x_max],
                color=BLUE,  # Changed to Blue
                stroke_width=4,
                name=f"GB Model {i}"
            )

            # Animate GB transformation next
            self.play(
                Transform(g_gb_graph, new_g_gb_graph),
                run_time=1.0  # Adjust run_time as needed
            )
            self.wait(0.1)  # Brief pause

            # Update GB variable label
            if np.isclose(beta_gb_new[1], 0):
                gb_new_text = f"{beta_gb_new[0]:.2f}"
            else:
                gb_new_text = f"{beta_gb_new[0]:.2f} + {beta_gb_new[1]:.2f}x"

            gb_new_variable_label = MathTex(
                gb_new_text,
                color=BLUE,  # Changed to Blue
                font_size=36
            )
            gb_new_variable_label.next_to(gb_fixed_label, RIGHT, buff=0.1)  # Position next to fixed label

            # Animate GB label transformation
            self.play(
                ReplacementTransform(gb_variable_label, gb_new_variable_label),
                run_time=0.5  # Fast update
            )

            # Update the GB variable label reference
            gb_variable_label = gb_new_variable_label

            # -----------------------------
            # C. Update Iteration Number
            # -----------------------------
            iteration_text_new = MathTex(rf"\text{{Iteration: }} {i}", font_size=50).set_color(BLACK).to_corner(UL, buff=0.5)
            self.play(
                ReplacementTransform(iteration_text, iteration_text_new),
                run_time=0.2
            )
            iteration_text = iteration_text_new

            self.wait(0.1)  # Brief pause

            # -----------------------------
            # D. Update Parameters for Next Iteration
            # -----------------------------
            beta_gd = beta_gd_new
            beta_gb = beta_gb_new

        # -----------------------------
        # 8. Final Pause
        # -----------------------------
        self.wait(2)  # Keep the final scene on screen for 2 seconds
