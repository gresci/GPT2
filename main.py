import json

import ai_integration
import tensorflow as tf

from inputs import openwebtext, openwebtext_longbiased, openwebtext_long
from model_fns import gpt2_model
from functools import partial

from models.gpt2 import encoder

from inputs import gpt2_pred_input


inputs = {
    "openwebtext": openwebtext,  # Standard OpenWebtext input
    "openwebtext_longbiased": openwebtext_longbiased,
    # OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": openwebtext_long,  # Openwebtext that only shows long examples
}

predict_mode = True

# Read params of model
with open("PrettyBig.json", "r") as f:
    params = json.load(f)

params["use_tpu"] = False

params["top_k"] = 100  # This controls text gen size

if not "precision" in params.keys():
    params[
        "precision"] = "float32"  # Doesn't actually do anything since float32 is the default anyways. Only recognized other dtype is "bfloat16"

if not "iterations" in params.keys():
    params["iterations"] = 1  # Because this controls how many samples are prefetched

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
    model_fn=gpt2_model,
    config=run_config,
    params=params)

enc = encoder.get_encoder(params["encoder_path"])

while True:
    with ai_integration.get_next_input(inputs_schema={"text": {"type": "text"}}) as inputs_dict:
        # If an exception happens in this 'with' block, it will be sent back to the ai_integration library
        predictions = network.predict(input_fn=partial(gpt2_pred_input, text=inputs_dict['text']))

        p = predictions[0]  # return just the first one
        p = p["tokens"]
        result_text = enc.decode(p)

        result_data = {
            "content-type": 'text/plain',
            "data": result_text,
            "success": True
        }
        ai_integration.send_result(result_data)

# Train eval loop
# input_fn = inputs[params["input"]]

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
