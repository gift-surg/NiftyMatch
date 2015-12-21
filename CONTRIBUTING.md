Please follow the steps in the guidelines below (you can find an example following the guidelines). If in doubt, ask Dzhoshkun Shakir (`d.shakir@ucl.ac.uk`).

# Guidelines

1. Create a new issue.
1. Branch off `dev` for the new issue.
1. Define your tests (usually in the `NiftyMatch-Test` repository).
1. Do your coding.
1. __TEST YOUR IMPLEMENTATION__.
1. __MAKE SURE CONTINUOUS INTEGRATION BUILDS OF YOUR BRANCH ARE OK__.
1. Submit a pull request from the new branch to `dev`.

# Example

1. When you add a new issue numbered `19` and entitled `Add support for interpolation`,
1. branch off `dev` to a new branch called `19-add-support-for-interpolation`.
1. Think about how you will test the interpolation method you'll implement. If needed, add automated tests into `NiftyMatch-Test`.
1. Implement the interpolation method.
1. __TEST YOUR IMPLEMENTATION__.
1. __MAKE SURE CONTINUOUS INTEGRATION BUILDS OF `19-add-support-for-interpolation` ARE OK__.
1. Submit a pull request from `19-add-support-for-interpolation` to `dev`.
