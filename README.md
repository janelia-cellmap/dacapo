# Dacapo

## Installation:
1) clone this repository
2) Some of our libraries have dependencies we need to install first (`numpy` and `cython`)
so, `pip install -r build-requirements.txt`
3) `pip install -r requirements.txt`
4) `pip install .`

## Usage:
### Web interface:
Install the [DaCapo dashboard](https://github.com/pattonw/dacapo-dashboard)
run `dacapo-dashboard dashboard`

### Command line interface:
The dashboard is basically just a user interface, you can do everything manually if you want.

#### Defining your configs:
Usually you would do this on the dashboard, but if you want you can simply import
any of the configurable classes of DaCapo and initialize your own version.

Here's an example python script that would create a new optimizer and save it to mongodb.
```python
from dacapo.optimizers import Optimizer, Adam
from dacapo.store import MongoDbStore

my_new_optimizer = Optimizer(name="my_new_adam", batch_size=5, algorithm=Adam(lr=0.001))
store = MongoDbStore()  # you must be working in a directory with a dacapo.yaml file

optimizer_id = store.add_optimizer(my_new_optimizer)
```
From now on you can use your optimizer by simply providing the `optimizer_id`.
Whenever DaCapo needs the optimizer, it will just call `store.get_optimizer(optimizer_id)`

For documentation on configurable classes please read the following READMEs:
1) "dacapo/tasks/README.md"
2) "dacapo/datasets/README.md"
3) "dacapo/models/README.md"
4) "dacapo/optimizers/README.md"
5) "dacapo/configs/README.md"

#### Training:
Once you have stored a `Run` config in the MongoDB. Training is as simple as
`dacapo run-one -r {your-run-id-here}`.
Everything related to how to train/validate is stored in the MongoDb.

Training will produce some outputs. You can explore the directory
"runs/{your-run-id-here}" to monitor your run. In there you will find snapshots,
evaluations, model-checkpoints, and logs.

#### Validations:
To Validate your run from the command line, simply call
`dacapo validate-one -r {your-run-id-here} -i {iteration}`.

This is the same call that is made internally. Please note this will only work
if you have already trained a model up to your chosen iteration, and your chosen
iteration aligns with the validation_interval specified in you `Run` config.
This is due to the fact that we only write out checkpoints on the validation interval.

#### Prediction:
Prediction needs a few more arguments since you will sometimes want to predict
on arbitrary data, and thus we cannot infer all the inputs. More details are available
with `dacapo predict-one --help`

WARNING: This is currently broken. Getting a "stream closed" tornado error from daisy.