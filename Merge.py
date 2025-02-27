from manim import *
import numpy as np
from sklearn.neighbors import KNeighborsRegressor  # Import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


class GradientDescentAndBoostingPlot(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"  # Custom background color (light beige)

    def construct(self):
        # -----------------------------
        # 1. Define the true function f(x)
        # -----------------------------
        def f(x):
            return 1 + 2 * x

        # -----------------------------
        # 2. Set random seed for reproducibility
        # -----------------------------
        np.random.seed(42)  # For consistent results

        # -----------------------------
        # 3. Generate data points
        # -----------------------------
        num_points = 500  # Number of data points
        x_min, x_max = -1, 1  # Range of x values

        # Uniformly distributed x values
        x_values = np.random.uniform(x_min, x_max, num_points)

        # Compute y values with added Gaussian noise
        noise_std = 0.1  # Standard deviation of noise
        y_values = f(x_values) + np.random.normal(0, noise_std, num_points)

        # Define outliers at indices 200 and 300
        outlier_indices = [200, 300]
        # y_values[outlier_indices[0]] = -0.5  # Example outlier value for point 200
        # y_values[outlier_indices[1]] = 2.5  # Example outlier value for point 300

        # -----------------------------
        # 4. Create axes
        # -----------------------------
        axes = Axes(
            x_range=[x_min-0.2, x_max+0.2, 0.5],
            y_range=[-1.5, 3.5, 0.5],  # Adjusted y_range to accommodate noise
            axis_config={"include_numbers": True},
            tips=True,  # Include arrow tips
        ).set_color(BLACK)

        # Add labels to the axes
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set(color=BLACK)

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
        # outlier_points = [points[i] for i in outlier_indices]

        # Create a Dot for each regular point with a smaller radius and orange color
        dots = VGroup(*[
            Dot(point=point, radius=0.02, color=ORANGE)  # Regular points
            for point in regular_points
        ])

        # -----------------------------
        # 6. Create Outliers
        # -----------------------------
        def create_outliers(outlier_pts):
            """
            Creates Dot objects for outliers.

            Parameters:
                outlier_pts (list): List of Manim points representing outliers.

            Returns:
                VGroup: A group of outlier dots.
            """
            outlier_dots = VGroup(*[
                Dot(point=point, radius=0.1, color=RED)  # Larger radius and red color
                for point in outlier_pts
            ])
            return outlier_dots

        # # Initialize outliers (not yet added to the scene)
        # outliers = create_outliers(outlier_points)

        # -----------------------------
        # 7. Add axes and labels to the scene
        # -----------------------------
        self.add(axes, axes_labels)  # Add axes and labels without animation
        self.wait(0.5)  # Brief pause before plotting points

        # -----------------------------
        # 8. Animate plotting the regular points and outliers over 1 second
        # -----------------------------
        # Use the FadeIn animation with run_time=1 second for better visibility
        # self.play(FadeIn(dots), FadeIn(outliers), run_time=1)
        # self.wait(0.5)  # Brief pause after plotting points
        self.play(FadeIn(dots), run_time=1)
        self.wait(0.5)  # Brief pause after plotting points

        # -----------------------------
        # 9. Plot the true function f(x) as a thin orange line without label
        # -----------------------------
        true_line = axes.plot(
            lambda x: f(x),
            color=ORANGE,
            stroke_width=1,
            x_range=[x_min, x_max]
        )
        self.play(Create(true_line), run_time=0.5)
        self.wait(0.5)  # Brief pause after plotting the true line

        # -----------------------------
        # 10. Set up Gradient Descent
        # -----------------------------
        # Define the degree of the polynomial for Gradient Descent
        degree_gd = 1  # Linear model

        # Initialize beta coefficients for Gradient Descent:
        # Set beta_0 to 1 and the rest to 0
        beta_gd = np.zeros(degree_gd + 1)
        beta_gd[0] = 1  # Initialize beta_0 to 1

        # Define learning rate for Gradient Descent
        learning_rate_gd = 0.375

        # Number of iterations for Gradient Descent
        num_iterations_gd = 10

        # Precompute powers of x for efficiency in Gradient Descent
        X_gd = np.vstack([x_values ** i for i in range(degree_gd + 1)]).T  # Shape: (num_points, degree_gd+1)

        # Compute gradients and store beta for each iteration in Gradient Descent
        betas_history_gd = [beta_gd.copy()]
        losses_history_gd = []
        losses_relative_gd = []  # To store MSE relative to f(x)

        for it in range(num_iterations_gd):
            # Compute current predictions
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

        # -----------------------------
        # 11. Set up Gradient Boosting with k-NN Regressor
        # -----------------------------
        # Define the number of boosting iterations
        num_iterations_gb = 10

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

        for it in range(num_iterations_gb):
            # Compute residuals
            residuals_gb = y_values - cumulative_predictions_gb

            # Initialize and fit the k-NN Regressor with k=5
            model_gb = KNeighborsRegressor(n_neighbors=5)
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

        # -----------------------------
        # 12. Prepare for Animation of Gradient Descent and Boosting
        # -----------------------------
        # Create labels for Gradient Descent
        iteration_label_gd = MathTex("GD \\ Iteration: \\ 0").to_corner(UL).set_color(ORANGE)
        loss_label_gd = MathTex(f"GD \\ MSE: \\ {losses_relative_gd[0]:.6f}").next_to(iteration_label_gd,
                                                                                      DOWN).set_color(ORANGE)

        # Create labels for Gradient Boosting
        iteration_label_gb = MathTex("GB \\ Iteration: \\ 0").to_corner(UR).set_color(BLUE)
        loss_label_gb = MathTex(f"GB \\ MSE: \\ {gb_losses_relative[0]:.6f}").next_to(iteration_label_gb,
                                                                                      DOWN).set_color(BLUE)

        self.add(iteration_label_gd, loss_label_gd, iteration_label_gb, loss_label_gb)

        # Define a fixed set of x values for plotting polynomials
        x_plot = np.linspace(x_min, x_max, 1000)

        # Define a function to create the polynomial graph for Gradient Descent
        def get_polynomial_gd(beta_coeffs, color=ORANGE):
            return axes.plot(
                lambda x: sum(beta_coeffs[j] * x ** j for j in range(len(beta_coeffs))),
                color=color,
                x_range=[x_min, x_max]
            )

        # Define a function to create the Gradient Boosting cumulative model
        def get_polynomial_gb(preds, color=BLUE):
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
        current_poly_gd = get_polynomial_gd(betas_history_gd[0], color=ORANGE)
        self.play(Create(current_poly_gd), run_time=0.5)

        # Initialize the first Gradient Boosting polynomial (initial prediction)
        current_poly_gb = get_polynomial_gb(cumulative_predictions_history_gb[0], color=BLUE)
        self.play(Create(current_poly_gb), run_time=0.5)

        # Initialize the equation label for Gradient Descent
        beta0 = betas_history_gd[0][0]
        beta1 = betas_history_gd[0][1]
        gd_equation_label = MathTex(f"y = {beta0:.2f} + {beta1:.2f}x").set_color(ORANGE).to_corner(DL)
        self.add(gd_equation_label)

        # Define the number of animation steps
        num_frames = max(num_iterations_gd, num_iterations_gb)

        for it in range(1, num_frames + 1):
            # Update Gradient Descent if within its iteration count
            if it <= num_iterations_gd:
                new_beta_gd = betas_history_gd[it]
                new_poly_gd = get_polynomial_gd(new_beta_gd, color=ORANGE)

                # Update labels for Gradient Descent
                iteration_label_gd_new = MathTex(f"GD \\ Iteration: \\ {it}").to_corner(UL).set_color(ORANGE)
                loss_label_gd_new = MathTex(f"GD \\ MSE: \\ {losses_relative_gd[it - 1]:.6f}").next_to(
                    iteration_label_gd_new, DOWN).set_color(ORANGE)

                # Update the equation label for Gradient Descent
                beta0 = new_beta_gd[0]
                beta1 = new_beta_gd[1]
                gd_equation_label_new = MathTex(f"y = {beta0:.2f} + {beta1:.2f}x").set_color(ORANGE).to_corner(DL)

                # Animate Gradient Descent model update and label updates
                self.play(
                    Transform(current_poly_gd, new_poly_gd),
                    Transform(iteration_label_gd, iteration_label_gd_new),
                    Transform(loss_label_gd, loss_label_gd_new),
                    Transform(gd_equation_label, gd_equation_label_new),
                    run_time=0.7
                )

            # Update Gradient Boosting if within its iteration count
            if it <= num_iterations_gb:
                # Get the cumulative predictions for the current iteration
                gb_current_pred = cumulative_predictions_history_gb[it]

                # Create the new Gradient Boosting polynomial
                new_poly_gb = get_polynomial_gb(gb_current_pred, color=BLUE)

                # Update labels for Gradient Boosting
                iteration_label_gb_new = MathTex(f"GB \\ Iteration: \\ {it}").to_corner(UR).set_color(BLUE)
                loss_label_gb_new = MathTex(f"GB \\ MSE: \\ {gb_losses_relative[it - 1]:.6f}").next_to(
                    iteration_label_gb_new, DOWN).set_color(BLUE)

                # Animate Gradient Boosting model update
                self.play(
                    Transform(current_poly_gb, new_poly_gb),
                    Transform(iteration_label_gb, iteration_label_gb_new),
                    Transform(loss_label_gb, loss_label_gb_new),
                    run_time=0.7
                )

            # Optionally, add a small wait to emphasize the update
            self.wait(0.2)

        self.wait(1)  # Pause at the end of the animation

