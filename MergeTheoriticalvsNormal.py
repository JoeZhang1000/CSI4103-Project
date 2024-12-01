from manim import *
import numpy as np
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientDescentAndBoostingPlot(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"  # Custom background color (light beige)

    def construct(self):
        # -----------------------------
        # Define Custom Colors
        # -----------------------------
        NAVY_BLUE = "#000080"  # Navy Blue Hex Code
        CUSTOM_BLUE = BLUE  # Dodger Blue Hex Code for replacement
        ORANGE_COLOR = ORANGE  # Orange for data points and f(x)
        BLACK_COLOR = BLACK

        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return np.random.randint(0, 2, size=x.shape)
            # Alternatively, using np.random.choice:
            # return np.random.choice([0, 1], size=x.shape, p=[0.5, 0.5])

        # -----------------------------
        # 2. Set random seed for reproducibility
        # -----------------------------
        np.random.seed(42)  # For consistent results

        # -----------------------------
        # 3. Generate data points
        # -----------------------------
        num_points = 100  # Number of data points
        x_min, x_max = -1, 1  # Range of x values

        # Uniformly distributed x values with slight noise
        x_values = np.linspace(x_min, x_max, num_points) + np.random.normal(0, 0.005, num_points)

        # Compute y values with added Gaussian noise
        noise_std = 0.1  # Standard deviation of noise
        y_values = f(x_values) + np.random.normal(0, noise_std, num_points)

        # Define outliers at indices 20 and 30
        outlier_indices = [20, 30]  # Adjusted for num_points=100 to prevent index error

        # -----------------------------
        # 4. Create axes
        # -----------------------------
        axes = Axes(
            x_range=[x_min - 0.2, x_max + 0.2, 0.5],
            y_range=[-1.5, 3.5, 0.5],  # Adjusted y_range to accommodate noise
            axis_config={"include_numbers": True},
            tips=True,  # Include arrow tips
        ).set_color(BLACK_COLOR)

        # Add labels to the axes
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set_color(BLACK_COLOR)

        axes.center()  # Center the axes in the scene

        # -----------------------------
        # 5. Create dots for each data point
        # -----------------------------
        # Convert data points to Manim's coordinate system
        points = [
            axes.c2p(x, y) for x, y in zip(x_values, y_values)
        ]

        # Separate regular points and outliers
        regular_points = [points[i] for i in range(num_points) if i not in outlier_indices]

        # Create a Dot for each regular point with a smaller radius and orange color
        dots = VGroup(*[
            Dot(point=point, radius=0.028, color=ORANGE_COLOR)  # Regular points
            for point in regular_points
        ]).set_z_index(10)  # Ensure dots are on top

        # -----------------------------
        # 6. Add axes and labels to the scene
        # -----------------------------
        self.add(axes, axes_labels)  # Add axes and labels without animation
        self.wait(0.5)  # Brief pause before plotting points

        # -----------------------------
        # 7. Animate plotting the regular points over 1 second
        # -----------------------------
        self.play(FadeIn(dots), run_time=1)
        self.wait(0.5)  # Brief pause after plotting points

        # -----------------------------
        # 8. Plot the true function f(x) as a thin orange line without label
        # -----------------------------
        true_line = axes.plot(
            lambda x: f(x),
            color=ORANGE_COLOR,
            stroke_width=1,
            x_range=[x_min, x_max]
        )
        true_line.set_z_index(1)  # Ensure true_line is below dots
        self.play(Create(true_line), run_time=0.5)
        self.wait(0.5)  # Brief pause after plotting the true line

        # -----------------------------
        # 9. Set up Gradient Descent
        # -----------------------------
        # Define the degree of the polynomial for Gradient Descent
        degree_gd = 2  # Degree 2 polynomial

        # Initialize beta coefficients for Gradient Descent:
        # Set beta_0 to the mean of y_values to align with GB's starting point
        beta_gd = np.zeros(degree_gd + 1)
        beta_gd[0] = y_values.mean()  # Initialize beta_0 to y_values.mean()

        # Define learning rate for Gradient Descent
        learning_rate_gd = 0.375

        # Number of iterations for Gradient Descent
        num_iterations_gd = 10000  # Total iterations for Gradient Descent

        # Precompute powers of x for efficiency in Gradient Descent
        X_gd = np.vstack([x_values ** i for i in range(degree_gd + 1)]).T  # Shape: (num_points, degree_gd+1)

        # Compute gradients and store beta for each iteration in Gradient Descent
        betas_history_gd = [beta_gd.copy()]
        losses_history_gd = []
        losses_relative_gd = []  # To store MSE relative to f(x)

        # -----------------------------
        # 10. Set up Gradient Boosting with Decision Stump
        # -----------------------------
        # Define the number of boosting iterations
        num_iterations_gb = 10000  # Extended to 10,000 iterations

        # Define learning rate for Gradient Boosting
        learning_rate_gb = 0.25

        # Initialize Gradient Boosting parameters
        # Initialize cumulative predictions as the mean of y
        cumulative_predictions_gb = np.full_like(y_values, y_values.mean())

        # Store cumulative predictions and losses
        cumulative_predictions_history_gb = [cumulative_predictions_gb.copy()]
        gb_losses = [mean_squared_error(y_values, cumulative_predictions_gb)]  # Original MSE
        gb_losses_relative = [mean_squared_error(f(x_values), cumulative_predictions_gb)]  # Relative MSE

        # Initialize list to store base learners
        base_learners_gb = []

        # -----------------------------
        # 11. Add Iteration Number, Loss Displays, and Legend to the Scene
        # -----------------------------
        # Create iteration number text in the top left corner with larger font size
        iteration_text = MathTex(r"\text{Iteration: } 0", font_size=50).set_color(BLACK_COLOR).to_corner(UL, buff=0.5)
        self.add(iteration_text)

        # Create GD Loss display
        gd_loss_text = MathTex(r"GD\ Loss: ", "0.0000", font_size=36).next_to(iteration_text, DOWN, buff=0.3).set_color(NAVY_BLUE)
        gd_loss_text[1].set_color(NAVY_BLUE)  # Set only the loss value to NAVY_BLUE

        # Create GB Loss display
        gb_loss_text = MathTex(r"GB\ Loss: ", "0.0000", font_size=36).next_to(gd_loss_text, DOWN, buff=0.2).set_color(CUSTOM_BLUE)
        gb_loss_text[1].set_color(CUSTOM_BLUE)  # Set only the loss value to CUSTOM_BLUE

        self.add(gd_loss_text, gb_loss_text)

        # Legend for f(x)
        f_label = MathTex(r"f(x)", color=ORANGE_COLOR, font_size=36)  # Increased font size
        f_label_group = VGroup(Dot(color=ORANGE_COLOR, radius=0.07), f_label).arrange(RIGHT, buff=0.2)

        # Legend for GD (now using NAVY_BLUE)
        gd_fixed_label = MathTex(r"g_1(x)", color=NAVY_BLUE, font_size=36).set_color(NAVY_BLUE)  # Updated color
        gd_label_group = VGroup(Dot(color=NAVY_BLUE, radius=0.07), gd_fixed_label).arrange(RIGHT, buff=0.1)

        # Legend for GB (now using CUSTOM_BLUE)
        gb_fixed_label = MathTex(r"g_2(x)", color=CUSTOM_BLUE, font_size=36).set_color(CUSTOM_BLUE)  # Updated color
        gb_label_group = VGroup(Dot(color=CUSTOM_BLUE, radius=0.07), gb_fixed_label).arrange(RIGHT, buff=0.1)

        # Combine all legends
        legend = VGroup(f_label_group, gd_label_group, gb_label_group).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        legend.to_corner(DOWN + RIGHT, buff=0.5)  # Move legend closer to the bottom-right corner
        self.add(legend)

        # -----------------------------
        # 12. Prepare for Animation of Gradient Descent and Boosting
        # -----------------------------
        # Define a fixed set of x values for plotting polynomials
        x_plot = np.linspace(x_min, x_max, 1000)

        # Define a function to create the polynomial graph for Gradient Descent
        def get_polynomial_gd(beta_coeffs, color=NAVY_BLUE):
            return axes.plot(
                lambda x: sum(beta_coeffs[j] * x ** j for j in range(len(beta_coeffs))),
                color=color,
                x_range=[x_min, x_max]
            )

        # Define a function to create the Gradient Boosting cumulative model
        def get_polynomial_gb(preds, color=CUSTOM_BLUE):
            # Sort the x and predictions for smooth plotting
            sorted_indices = np.argsort(x_values)
            sorted_x = x_values[sorted_indices]
            sorted_preds = preds[sorted_indices]
            return axes.plot_line_graph(
                x_values=sorted_x,
                y_values=sorted_preds,
                add_vertex_dots=False,
                line_color=color,
            )

        # Initialize the first Gradient Descent polynomial (initial beta)
        current_poly_gd = get_polynomial_gd(betas_history_gd[0], color=NAVY_BLUE)
        current_poly_gd.set_z_index(2)  # Ensure GD line is below dots
        self.play(Create(current_poly_gd), run_time=0.5)

        # Initialize the first Gradient Boosting polynomial (initial prediction)
        current_poly_gb = get_polynomial_gb(cumulative_predictions_history_gb[0], color=CUSTOM_BLUE)
        current_poly_gb.set_z_index(2)  # Ensure GB line is below dots
        self.play(Create(current_poly_gb), run_time=0.5)

        # -----------------------------
        # 13. Animate the First 10 Iterations with Loss Displays
        # -----------------------------
        for it in range(1, 11):
            # -----------------------------
            # Gradient Descent Iteration
            # -----------------------------
            # Compute current predictions for GD
            predictions_gd = X_gd.dot(beta_gd)  # Shape: (num_points,)

            # Compute errors for training (relative to y_values)
            errors_gd = predictions_gd - y_values  # Shape: (num_points,)

            # Compute loss (Mean Squared Error) for training
            loss_gd = np.mean(errors_gd ** 2)
            losses_history_gd.append(loss_gd)

            # Compute loss relative to true function f(x)
            loss_gd_relative = np.mean((predictions_gd - f(x_values)) ** 2)
            losses_relative_gd.append(loss_gd_relative)

            # Compute gradients
            gradients_gd = (2 / num_points) * X_gd.T.dot(errors_gd)  # Shape: (degree_gd+1,)

            # Update beta coefficients
            beta_gd = beta_gd - learning_rate_gd * gradients_gd

            # Store the updated beta
            betas_history_gd.append(beta_gd.copy())

            # Update GD loss display
            new_gd_loss = f"{loss_gd:.4f}"
            new_gd_loss_text = MathTex(r"GD\ Loss: ", new_gd_loss, font_size=36).set_color(NAVY_BLUE)
            new_gd_loss_text[1].set_color(NAVY_BLUE)
            self.play(Transform(gd_loss_text, new_gd_loss_text), run_time=0.3)

            # -----------------------------
            # Gradient Boosting Iteration
            # -----------------------------
            # Compute residuals
            residuals_gb = y_values - cumulative_predictions_gb

            # Initialize and fit the Decision Stump (DecisionTreeRegressor with max_depth=1)
            model_gb = DecisionTreeRegressor(max_depth=1, random_state=42)
            model_gb.fit(x_values.reshape(-1, 1), residuals_gb)

            # Predict residuals
            preds_gb = model_gb.predict(x_values.reshape(-1, 1))

            # Update cumulative predictions
            cumulative_predictions_gb += learning_rate_gb * preds_gb

            # Store the base learner
            base_learners_gb.append(model_gb)

            # Compute original loss
            loss_gb = mean_squared_error(y_values, cumulative_predictions_gb)
            gb_losses.append(loss_gb)

            # Compute relative loss
            loss_gb_relative = mean_squared_error(f(x_values), cumulative_predictions_gb)
            gb_losses_relative.append(loss_gb_relative)

            # Store cumulative predictions
            cumulative_predictions_history_gb.append(cumulative_predictions_gb.copy())

            # Update GB loss display
            new_gb_loss = f"{loss_gb:.4f}"
            new_gb_loss_text = MathTex(r"GB\ Loss: ", new_gb_loss, font_size=36).set_color(CUSTOM_BLUE)
            new_gb_loss_text[1].set_color(CUSTOM_BLUE)
            self.play(Transform(gb_loss_text, new_gb_loss_text), run_time=0.3)

            # -----------------------------
            # Update Iteration Number
            # -----------------------------
            new_iteration_text = MathTex(rf"\text{{Iteration: }} {it}", font_size=50).set_color(BLACK_COLOR).to_corner(UL, buff=0.5)
            self.play(Transform(iteration_text, new_iteration_text), run_time=0.3)

            # -----------------------------
            # Update Gradient Descent Polynomial
            # -----------------------------
            new_poly_gd = get_polynomial_gd(beta_gd, color=NAVY_BLUE)
            new_poly_gd.set_z_index(2)  # Ensure GD line is below dots

            # Animate Gradient Descent model update
            self.play(
                Transform(current_poly_gd, new_poly_gd),
                run_time=0.5
            )

            # -----------------------------
            # Update Gradient Boosting Polynomial
            # -----------------------------
            gb_current_pred = cumulative_predictions_gb
            new_poly_gb = get_polynomial_gb(gb_current_pred, color=CUSTOM_BLUE)
            new_poly_gb.set_z_index(2)  # Ensure GB line is below dots

            # Animate Gradient Boosting model update
            self.play(
                Transform(current_poly_gb, new_poly_gb),
                run_time=0.5
            )

            self.wait(0.1)

        # -----------------------------
        # 14. Show the Gradient Boosting Line after 10,000 Iterations
        # -----------------------------
        # Create the final GB line after 10,000 iterations
        final_poly_gb = get_polynomial_gb(cumulative_predictions_gb, color=CUSTOM_BLUE)
        final_poly_gb.set_stroke(width=3)
        final_poly_gb.set_z_index(2)  # Ensure GB line is below dots

        # Animate the drawing of the final GB line without removing the previous one
        self.play(
            Create(final_poly_gb),
            run_time=1
        )

        # Update GB loss display to final loss
        final_gb_loss = f"{gb_losses[-1]:.4f}"
        final_gb_loss_text = MathTex(r"GB\ Loss: ", final_gb_loss, font_size=36).set_color(CUSTOM_BLUE)
        final_gb_loss_text[1].set_color(CUSTOM_BLUE)
        self.play(Transform(gb_loss_text, final_gb_loss_text), run_time=0.3)

        self.wait(1)  # Final pause

        # -----------------------------
        # 15. Show the Gradient Descent Line after 10,000 Iterations and Remove the 10-Iteration Line
        # -----------------------------
        # Create the final GD line after 10,000 iterations
        final_poly_gd = get_polynomial_gd(beta_gd, color=NAVY_BLUE)
        final_poly_gd.set_stroke(width=3)
        final_poly_gd.set_z_index(2)  # Ensure GD line is below dots

        # Animate the drawing of the final GD line and remove the 10-iteration line
        self.play(
            Create(final_poly_gd),
            FadeOut(current_poly_gd),
            run_time=1
        )

        # Update GD loss display to final loss
        final_gd_loss = f"{loss_gd:.4f}"
        final_gd_loss_text = MathTex(r"GD\ Loss: ", final_gd_loss, font_size=36).set_color(NAVY_BLUE)
        final_gd_loss_text[1].set_color(NAVY_BLUE)
        self.play(Transform(gd_loss_text, final_gd_loss_text), run_time=0.3)

        self.wait(1)  # Final pause

        # -----------------------------
        # 16. Overlay the Orange Points on Top of All Lines
        # -----------------------------
        # Since dots already have a higher z_index, ensure no other objects have higher z_index
        # Alternatively, re-add the dots to ensure they're on top
        self.add(dots)  # This will bring dots to the top layer

        self.wait(2)  # Final pause to observe the overlaid points
