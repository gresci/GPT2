# GPT2 Text Generator


## For details see [Text Generator](https://deepai.org/machine-learning-model/text-generator) on [Deep AI](https://deepai.org).

<p>
    <a href="https://cloud.docker.com/u/deepaiorg/repository/docker/deepaiorg/gpt2">
        <img src='https://img.shields.io/docker/cloud/automated/deepaiorg/gpt2.svg?style=plastic' />
        <img src='https://img.shields.io/docker/cloud/build/deepaiorg/gpt2.svg' />
    </a>
</p>

This model has been integrated with [ai_integration](https://github.com/deepai-org/ai_integration/blob/master/README.md) for seamless portability across hosting providers.


# Quick Start
```bash
docker pull deepaiorg/gpt2
```

### HTTP
```bash
docker run --rm -it -e MODE=http -p 5000:5000 deepaiorg/gpt2
```
Open your browser to localhost:5000 (or the correct IP address)

### Command Line
Save your input text as input.txt in the current directory.
```bash
docker run --rm -it -v `pwd`:/shared -e MODE=command_line deepaiorg/gpt2 --text /shared/input.txt
```

### Command Line with Piped input

```bash
echo '{"text":"I am very happy because this model is great!"}' | docker run -e MODE=test_inputs_dict_json --rm -i deepaiorg/gpt2
```

# Docker build
```bash
docker build -t gpt2 .
```

# Credit
Connor Leahy: https://github.com/ConnorJL/GPT2
GPT-2 Model by OpenAI
