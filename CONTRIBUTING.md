# Contributing to Dominus Ultra

We welcome contributions to `Dominus Ultra`! Your help is invaluable in making this high-performance Triton kernel even better. Please take a moment to review this document to understand our contribution guidelines.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please open an issue on our GitHub repository. When reporting a bug, please include:

-   A clear and concise description of the bug.
-   Steps to reproduce the behavior.
-   Expected behavior.
-   Actual behavior.
-   Your environment details (GPU, CUDA version, Triton version, PyTorch version).
-   Any relevant error messages or stack traces.

### 2. Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an issue to discuss your ideas. Clearly describe the enhancement and why you believe it would be beneficial to the project.

### 3. Submitting Pull Requests (PRs)

1.  **Fork the Repository**: Start by forking the `Dominus Ultra` repository to your GitHub account.
2.  **Clone Your Fork**: Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/DominusUltra.git
    cd DominusUltra
    ```
3.  **Create a New Branch**: Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    # or
    git checkout -b bugfix/your-bug-fix-name
    ```
4.  **Make Your Changes**: Implement your changes, ensuring they adhere to our coding standards (see below).
5.  **Test Your Changes**: Run existing tests and add new tests for your changes to ensure correctness and prevent regressions.
6.  **Commit Your Changes**: Write clear, concise commit messages. Follow the Conventional Commits specification (e.g., `feat: add new GQA support`, `fix: resolve FP8 precision issue`).
7.  **Push to Your Fork**: Push your new branch to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
8.  **Open a Pull Request**: Go to the original `Dominus Ultra` repository on GitHub and open a new Pull Request from your forked branch. Please provide a detailed description of your changes, including:
    -   What problem does this PR solve?
    -   How does it solve it?
    -   Any relevant benchmarks or performance improvements.
    -   Screenshots or code snippets if applicable.

## Coding Standards

-   **Python**: Adhere to PEP 8. Use `black` for formatting and `flake8` for linting.
-   **Triton**: Follow Triton best practices for kernel development, focusing on memory efficiency, parallelism, and avoiding bank conflicts.
-   **Documentation**: Ensure all new code is well-commented. Update `README.md` or `whitepaper.md` if your changes introduce new features or modify existing behavior.
-   **Testing**: Write unit tests for new functionalities. We use `pytest`.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Security Policy

If you discover a security vulnerability, please report it responsibly as described in our [Security Policy](SECURITY.md).

Thank you for contributing to `Dominus Ultra`!
