"""
    Modify it for configure the hyperparameters/experiment setup
"""
BATCH_SIZE = 4
EPOCHS = 100

IS_ADAPTER = False
IS_CURRICULUM = True

NAME = (
    "bart"
    + ("_adapter" if IS_ADAPTER else "_finetune")
    + ("_cur_modified_courses" if IS_CURRICULUM else "")
    + ("_no_cheat")
)

