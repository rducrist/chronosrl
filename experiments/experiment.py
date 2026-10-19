"""Training entry point: builds a Runner from the command-line flags and runs it."""

# Run like this
# pixi run -e crazyflow python experiments/experiment.py \
#     --config=configs/crazyflow_position.toml

from chronosrl.runner import Runner

if __name__ == "__main__":
    runner = Runner()
    runner.run()
