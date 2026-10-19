# Run like this 
# pixi run -e crazyflow python experiments/experiment.py --config=configs/crazyflow_position_hard.toml


from chronosrl.runner import Runner

if __name__ == "__main__":
    runner = Runner()
    runner.run()
