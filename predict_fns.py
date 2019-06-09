from functools import partial

from models.gpt2 import encoder

from inputs import gpt2_pred_input


# Takes in the user supplied text and generates output text. Returns text
def gpt2_predict(network, text, params):
    enc = encoder.get_encoder(params["encoder_path"])
    predictions = network.predict(input_fn=partial(gpt2_pred_input, text=text))

    for i, p in enumerate(predictions):
        p = p["tokens"]
        text = enc.decode(p)
        return text # return just the first one

