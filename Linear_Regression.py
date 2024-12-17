import numpy as np

def generate_data(rows=100, cols=5):
    """
    Simulate students' scores data where rows are students and cols are subjects.
    """
    np.random.seed(42)  # Reproducibility
    scores = np.random.randint(50, 100, size=(rows, cols))
    print(f"Generated Data ({rows}x{cols}):\n{scores}\n")
    return scores

def manipulate_data(data):
    """
    Showcase slicing, masking, broadcasting, and reshaping.
    """
    print("--- Data Manipulation ---")
    print("Original Shape:", data.shape)

    # Slicing: Get scores of first 5 students
    first_5 = data[:5, :]
    print("First 5 students' scores:\n", first_5)

    # Masking: Find students scoring >90 in any subject
    mask = data > 90
    print("\nMask (scores > 90):\n", mask)
    high_scorers = np.where(mask, data, np.nan)  # Replace non-matching values with NaN
    print("\nHigh scorers (scores > 90):\n", high_scorers)

    # Broadcasting: Add 5 bonus marks to all students in Subject 1
    data[:, 0] += 5
    print("\nAfter adding 5 bonus marks to Subject 1:\n", data)

    # Reshaping: Flatten data and reshape
    flattened = data.flatten()
    reshaped = flattened.reshape(-1, 5)
    print("\nReshaped Data (after flattening and reshaping):\n", reshaped)

def calculate_statistics(data):
    """
    Calculate basic statistics on the scores dataset.
    """
    print("--- Statistics ---")
    print("Mean Scores (per subject):", np.mean(data, axis=0))
    print("Median Scores (per subject):", np.median(data, axis=0))
    print("Standard Deviation (per subject):", np.std(data, axis=0))
    print("Overall Max Score:", np.max(data))
    print("Overall Min Score:", np.min(data))

def linear_regression_example():
    """
    Perform a basic linear regression using NumPy.
    Example: Predict scores based on study hours.
    """
    print("--- Linear Regression Example ---")
    np.random.seed(42)
    # Simulate study hours (X) and scores (Y)
    study_hours = np.random.randint(1, 10, size=50)
    scores = 5 * study_hours + np.random.randint(10, 20, size=50)  # Linear relation: Y = 5X + Noise

    print("Study Hours:\n", study_hours)
    print("Scores:\n", scores)

    # Linear regression: Solve for coefficients using least squares method
    X = np.vstack([study_hours, np.ones(len(study_hours))]).T  # Add bias term
    Y = scores
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]  # Solves for coefficients

    print(f"\nLinear Regression Equation: Y = {coefficients[0]:.2f}X + {coefficients[1]:.2f} (Intercept)")

def main():
    print("==== NumPy Concepts Showcase: Data Analysis App ====")

    # Step 1: Generate random data
    data = generate_data(rows=10, cols=5)  # 10 students, 5 subjects

    # Step 2: Perform data manipulations
    manipulate_data(data)

    # Step 3: Calculate statistics
    calculate_statistics(data)

    # Step 4: Perform Linear Regression Example
    linear_regression_example()

if __name__ == "__main__":
    main()
