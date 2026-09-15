# Contributing to GPD

Thank you for your interest in contributing to the GPD project! This document provides guidelines for contributing code, reporting issues, and proposing improvements.

---

## 🐛 Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title** describing the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs. actual behavior
4. **Environment details**:
   - Python version
   - PyTorch version
   - GPU/CPU
   - Operating system
5. **Error messages** or logs (if applicable)

---

## 💡 Suggesting Enhancements

We welcome suggestions for improvements! Please open an issue with:

1. **Clear description** of the proposed feature
2. **Motivation**: Why is this enhancement useful?
3. **Use case**: How would it be used?
4. **Possible implementation** (optional)

---

## 🔧 Code Contributions

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss major changes before implementing
3. **Fork the repository** and create a feature branch

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/GPD.git
cd GPD

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -r requirements-dev.txt
```

### Coding Standards

- **Python style**: Follow PEP 8
- **Code formatting**: Use `black` or `autopep8`
- **Type hints**: Add type annotations where appropriate
- **Documentation**: Add docstrings for new functions/classes
- **Comments**: Explain non-obvious logic

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Add comments where necessary
   - Update documentation if needed

3. **Test your changes**:
   - Ensure existing functionality still works
   - Add tests for new features (if applicable)

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a pull request**:
   - Provide a clear title and description
   - Reference related issues (e.g., "Fixes #123")
   - Explain what changes were made and why

### Code Review

- Maintainers will review your PR and may request changes
- Address feedback and push updates to your branch
- Once approved, your PR will be merged

---

## 📚 Documentation Contributions

Improvements to documentation are highly valued! You can:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation (if multilingual support is added)

---

## 🧪 Testing

If you add new functionality:

1. Write unit tests for new functions
2. Ensure tests pass before submitting PR
3. Update test data if necessary

```bash
# Run tests (if test suite exists)
pytest tests/
```

---

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment or discriminatory comments
- Trolling or insulting remarks
- Personal or political attacks
- Publishing others' private information

---

## 📝 Commit Message Guidelines

Use clear and descriptive commit messages:

- **Format**: `<type>: <subject>`
- **Types**:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation changes
  - `style`: Code formatting (no functional changes)
  - `refactor`: Code restructuring
  - `test`: Adding or updating tests
  - `chore`: Maintenance tasks

**Examples**:
- `feat: add attention visualization module`
- `fix: correct batch normalization in student model`
- `docs: update installation instructions for CUDA 12`

---

## 🙏 Attribution

Contributors will be acknowledged in:
- GitHub contributors list
- Release notes (for significant contributions)
- Project documentation

---

## ❓ Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Contact the maintainers directly

---

**Thank you for contributing to GPD!** 🎉
