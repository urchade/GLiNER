# A guide to making a release

This guide collects the steps we do in GLiNER to make a release on PyPI. They result from (variations of) hard-learned lessons and while following this guide is completely optional, it’s strongly recommended to do so. 🙂 This is a truncated version of the [SetFit](https://github.com/huggingface/setfit/blob/main/RELEASE.md) release guide, which is more exhaustive and does some additional steps.

### Preparation

To be able to make a release for a given project, you’ll need an account on [PyPI](https://pypi.org/) and on [Test PyPI](https://test.pypi.org/). If you are making a release for an existing project, your username will need to be added to that project by one of the current maintainers on PyPI. Note that we strongly recommend enabling two-factor authentication on PyPI.

You will also need to install twine in your Python environment with `pip install twine`.

Additionally, it can be nice to familiarize yourself with [Semantic Versioning](https://semver.org/). This is a fairly strict document, but it provides a useful summary that library maintainers should follow:

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backward compatible manner
> 3. PATCH version when you make backward compatible bug fixes
> 
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

The very first release should be "0.1.0".

## Releases

GLiNER supports two release methods: **automated releases via GitHub Actions** (recommended) and manual releases via `twine`. The automated method is preferred as it ensures consistency and uses PyPI's trusted publishing.

### Automated Release via GitHub Actions (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/release.yaml`) that automatically builds and publishes releases to PyPI when you push a version tag to the main branch.

#### Prerequisites

1. Ensure PyPI trusted publishing is configured for the GLiNER project in your PyPI account settings
2. The GitHub Actions workflow must have the proper PyPI environment configured

#### Steps for Automated Release

**Step 1: Adjust the version of your package**

Update the version in [`gliner/__init__.py`](gliner/__init__.py) from dev to release version:

```diff
- __version__ = "0.4.0.dev"
+ __version__ = "0.4.0"
```

Commit and push to main:

```bash
git add gliner
git commit -m "Release: v{VERSION}"
git push origin main
```

**Step 2: Create and push a version tag**

Create a tag matching the pattern `v*` (e.g., `v0.4.0`) and push it to trigger the release:

```bash
git tag v<VERSION>
git push origin v<VERSION>
```

**Important:** The tag MUST be pushed to the main branch. The workflow will automatically verify this and fail if the tag is from any other branch.

**Step 3: Monitor the GitHub Actions workflow**

1. Go to [https://github.com/urchade/GLiNER/actions](https://github.com/urchade/GLiNER/actions)
2. Watch the "Release GLiNER to PyPI" workflow execution
3. The workflow will:
   - Build the distribution packages (wheel and source tarball)
   - Verify the tag is on the main branch
   - Publish to PyPI using trusted publishing

**Step 4: (Optional) Prepare release notes**

Create release notes on GitHub at [https://github.com/urchade/GLiNER/releases](https://github.com/urchade/GLiNER/releases) using the tag you just created.

**Step 5: Bump the dev version**

After a successful release, update [`gliner/__init__.py`](gliner/__init__.py) to the next dev version:

```diff
- __version__ = "0.4.0"
+ __version__ = "0.4.1.dev"
```

Commit and push:

```bash
git add gliner
git commit -m "Bump version to {NEXT_VERSION}.dev"
git push origin main
```

### Manual Release via twine

If you need to release manually or the automated workflow is unavailable, follow these steps:

### Step 1: Adjust the version of your package

You should have the current version specified in [`gliner/__init__.py`](gliner/__init__.py). This version should be a dev version (e.g. `0.1.0.dev`) before you release, change it to the name of the version you are releasing:

```diff
- __version__ = "0.4.0.dev"
+ __version__ = "0.4.0"
```

Commit the changes on your release branch and push them:

```bash
git add gliner
git commit -m "Release: v{VERSION}"
git push -u origin main
```

### Step 2: (Optional) Make sure all tests pass

If you add tests, then you should also add CI, e.g. like this [`tests.yaml`](https://github.com/tomaarsen/SpanMarkerNER/blob/main/.github/workflows/tests.yaml) file. This will automatically run tests whenever you make changes, it can be very useful. Make sure all tests that you may have pass before proceeding to the next step.

### Step 3: Add a tag for your release

A tag will flag the exact commit associated to your release (and be easier to remember than the commit hash!). The tag should be `v<VERSION>` so for instance `v4.12.0`. 

Here is how you can create and push your tag:

```bash
git tag v<VERSION>
git push --tags origin main
```

### Step 4: (Optional) Prepare the release notes

You can then put your release notes in a Draft Release on GitHub, in [https://github.com/urchade/GLiNER/releases](https://github.com/urchade/GLiNER/releases) and write a small paragraph highlighting each of the new features this release is adding.

You can use the previously created tag to let GitHub auto-generate some release notes based on recent pull requests.

### Step 5: Create the wheels for your release

This is what you'll upload on PyPI and what everyone will download each time they `pip install` your package.

Clean previous builds by deleting the `build` and `dist` directories or by running:

```
rm -rf build && rm -rf dist
```

Then run:

```bash
python -m build
```

This will create two folders, `build` and a `dist` with the new versions of your package. These contain a 1) source distribution and a 2) wheel.

### Step 6: Upload your package on PyPI test

**DO NOT SKIP THIS STEP!**

This is the most important check before actually releasing your package in the wild. Upload the package on PyPI test and check you can properly install it.

To upload it:

```bash
twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
```

You will be prompted for your username and password. If that doesn't work, you can create an API Token for your Test PyPI account and create a `~/.pypirc` account if it doesn't already exist, with:

```
[distutils]
  index-servers =
    gliner_test

[gliner_test]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-...
```
(some more details on this [here](https://pypi.org/help/#apitoken))

And then run:
```bash
twine upload dist/* -r gliner_test
```

Once that has uploaded the package, in a fresh environment containing all dependencies you need (tip: you can use Google Colab for this!), try to install your new package from the PyPI test server. First install all dependencies, and then your package.

```bash
python -m pip install torch transformers huggingface_hub flair tqdm
python -m pip install -i https://testpypi.python.org/pypi gliner
```

If everything works, you should be able to run this code:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_base")

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

labels = ["person", "award", "date", "competitions", "teams"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

### Step 7: Publish on PyPI

This cannot be undone if you messed up, so make sure you have run Step 6!

Once you’re fully ready, upload your package on PyPI:

```bash
twine upload dist/* -r pypi
```

You will be prompted for your username and password, unless you're using the recommended [PyPI API token](https://pypi.org/help/#apitoken). 

### Step 8: (Optional) Publish your release notes

Go back to the draft you did at step 4 ([https://github.com/urchade/GLiNER/releases](https://github.com/urchade/GLiNER/releases)) and publish them.

### Step 9: Bump the dev version on the main branch

You’re almost done! Just go back to the `main` branch and change the dev version in [`gliner/__init__.py`](gliner/__init__.py) to the new version you’re developing, for instance `4.13.0.dev` if just released `4.12.0`.

**Note:** This step applies to both the automated and manual release processes.
