from manim import *
import numpy as np

class GradientDescentPlot(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = "#ddd7d0"  # Custom background color (light beige)

    def construct(self):
        # -----------------------------
        # 1. Define the function f(x)
        # -----------------------------
        def f(x):
            return 1+2*x

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
        y_values = f(x_values)

        # -----------------------------
        # 4. Create axes
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
        # 5. Create dots for each data point
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
        # 8. Set up Gradient Descent
        # -----------------------------
        # Define the degree of the polynomial
        degree = 1  # Changed from 10 to 4

        # Initialize beta coefficients:
        # Set beta_0 to 1 and the rest to 0
        beta = np.zeros(degree + 1)
        beta[0] = 1  # Initialize beta_0 to 1

        # Define learning rate
        learning_rate = 0.1  # Set to 0.001

        # Number of iterations
        num_iterations = 50  # Increased to 4000

        # Animation update frequency
        animation_interval = 5  # Animate every 500 iterations

        # Precompute powers of x for efficiency
        X = np.vstack([x_values**i for i in range(degree + 1)]).T  # Shape: (num_points, degree+1)

        # Compute gradients and store beta for each iteration
        betas_history = [beta.copy()]
        losses_history = []

        for it in range(num_iterations):
            # Compute current predictions
            predictions = X.dot(beta)  # Shape: (num_points,)

            # Compute errors
            errors = predictions - y_values  # Shape: (num_points,)

            # Compute loss (Mean Squared Error)
            loss = np.mean(errors**2)
            losses_history.append(loss)

            # Compute gradients
            gradients = (2 / num_points) * X.T.dot(errors)  # Shape: (degree+1,)

            # Update beta coefficients
            beta = beta - learning_rate * gradients

            # Store the updated beta
            betas_history.append(beta.copy())

        # -----------------------------
        # 9. Prepare for Animation of Gradient Descent
        # -----------------------------
        # Create labels for iteration and loss
        iteration_label = MathTex("Iteration: 0").to_corner(UL).set_color(BLACK)
        loss_label = MathTex("MSE: {:.6f}".format(losses_history[0])).next_to(iteration_label, DOWN).set_color(BLACK)
        self.add(iteration_label, loss_label)

        # Define a fixed set of x values for plotting polynomials
        x_plot = np.linspace(x_min, x_max, 1000)

        # Define a function to create the polynomial graph given beta coefficients
        def get_polynomial(beta_coeffs):
            # Compute y values based on the fixed x_plot
            y_plot = sum(beta_coeffs[j] * x_plot**j for j in range(degree + 1))
            # Return the graph without the 'resolution' parameter
            return axes.plot(
                lambda x: sum(beta_coeffs[j] * x**j for j in range(degree + 1)),
                color="#f4592f",
                x_range=[x_min, x_max]
            )

        # Initialize the first polynomial (beta = initial)
        current_poly = get_polynomial(betas_history[0])
        self.play(Create(current_poly), run_time=0.5)

        # Number of animation frames
        num_frames = num_iterations // animation_interval  # 4000 / 500 = 8

        # Prepare the list of iterations to animate
        animate_iterations = list(range(animation_interval, num_iterations + 1, animation_interval))

        for idx, it in enumerate(animate_iterations):
            new_beta = betas_history[it]
            new_poly = get_polynomial(new_beta)

            # Update labels
            iteration_label_new = MathTex(f"Iteration: {it}").to_corner(UL).set_color(BLACK)
            loss_label_new = MathTex(f"MSE: {losses_history[it-1]:.6f}").next_to(iteration_label_new, DOWN).set_color(BLACK)

            # Animate the transition:
            # - Fade out the old polynomial
            # - Fade in the new polynomial
            # - Update iteration and loss labels
            self.play(
                Transform(iteration_label, iteration_label_new),
                Transform(loss_label, loss_label_new),
                FadeOut(current_poly),
                Create(new_poly),
                run_time=2  # Total gradient descent animation time ~5.6 seconds (8 * 0.7)
            )

            # Update current_poly for next iteration
            current_poly = new_poly

        self.wait(1)  # Pause at the end of the animation

# Instructions to Render the Scene:
# 1. Save this script as `gradientDescent_plot.py`.
# 2. Open your terminal or command prompt and navigate to the directory containing the script.
# 3. Run the following command to render the scene:
#    manim -pql gradientDescent_plot.py GradientDescentPlot
#    - `-pql` stands for preview quality low. You can change it to `-pqh` for higher quality.
