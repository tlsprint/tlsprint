# TLSprint

Fingerprint TLS implementations using state machines. inferred by
[StateLearner](https://github.com/jderuiter/statelearner/). StateLearner can
learn state machines for (in this case TLS) implementations using a black-box
approach. Different implementations can result in a different state machine,
which makes it possible to differentiate them. By combining these state
machines into a single tree, and then probing a live implementation, `tlsprint`
makes it possible to fingerprint the TLS implementation running on the target.

## Installation

Install the latest release from PyPi:

```shell
pip install tlsprint`
```

## Learn

Note: This step is optional, a `model.p` is included in the distribution, which
contains a model created using 27 unique state machines, representing 283
different TLS implementations. For the full list of implementations, check the
`models` directory in the repository.

After state machines are inferred using StateLearner, run

```shell
tlsprint learn <statelearner_output_dir> model.p
```

to merge all models together into a single
tree. This tree is returned as a pickled `networkx` graph, and is required for
the `identify` step.

## Identify

When using the default model, identifying the TLS implementation on a target
can be done be running

```shell
tlsprint identify <target>
```

This defaults to port 443, but a custom port can be specified by adding
`--target-port <port>`.

The command returns a list of possible implementations. All these
implementations share the same model, meaning `tlsprint` cannot further specify
the exact implementation.

Passing `--graph-dir <output>` to the `identify` command, will write DOT files
for all intermediate versions of the model tree. This can be insightful to
understand what `tlsprint` is doing.

If you learned a custom model using the `learn` command, you can override the
default model using `--model <filename>`.
