import json
import sys

import tensorflow as tf
import ai_integration

from inputs import *
from model_fns import *
from predict_fns import *

# This program was designed to function with multiple kinds of models, but currently only GPT2 is supported
# The first element in the tupel is the model function, the second is the function called when predicting
models = {
    "GPT2": (gpt2_model, gpt2_predict)
}

inputs = {
    "openwebtext": openwebtext,  # Standard OpenWebtext input
    "openwebtext_longbiased": openwebtext_longbiased,
# OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": openwebtext_long,  # Openwebtext that only shows long examples
}

predict_mode = True

# Setup logging
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.StreamHandler(sys.stdout)
]
logger = logging.getLogger('tensorflow')
logger.handlers = handlers

# Read params of model
with open("PrettyBig.json", "r") as f:
    params = json.load(f)

params["use_tpu"] = False

params["top_k"] = 100 # This controls text gen size

if not "precision" in params.keys():
    params[
        "precision"] = "float32"  # Doesn't actually do anything since float32 is the default anyways. Only recognized other dtype is "bfloat16"

if not "iterations" in params.keys():
    params["iterations"] = 1  # Because this controls how many samples are prefetched


model_fn = models[params["model"]][0]
predict_fn = models[params["model"]][1]
input_fn = inputs[params["input"]]


# Non TPU setup
if not predict_mode:
    params["batch_size"] = params["train_batch_size"]
else:
    params["batch_size"] = params["predict_batch_size"]
run_config = tf.estimator.RunConfig(
    model_dir=params["model_path"],
    session_config=tf.ConfigProto(
        # log_device_placement=True,
        # allow_soft_placement=True
    ),
)

network = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params=params)

while True:
    with ai_integration.get_next_input(inputs_schema={"text": {"type": "text"}}) as inputs_dict:
        # If an exception happens in this 'with' block, it will be sent back to the ai_integration library

        result_text = predict_fn(network, inputs_dict['text'], params)
        result_data = {
            "content-type": 'text/plain',
            "data": result_text,
            "success": True
        }
        ai_integration.send_result(result_data)



# Train eval loop
# while True:
#     start = time.time()
#
#     network.train(
#         input_fn=partial(input_fn, eval=False),
#         steps=params["train_steps"])
#
#     end = time.time()
#     logger.info("\nTrain loop took {:.2f}s\n".format(end - start))
#
#     eval_result = network.evaluate(
#         input_fn=partial(input_fn, eval=True),
#         steps=params["eval_steps"])
#
#     logger.info("\nEval Results: {}\n".format(str(eval_result)))
#
#     if network.get_variable_value("global_step") > params["max_steps"]:
#         logger.info("Done!")
#         break
