import random
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Function Name : DisplayTitle
# Description   : It displays the project title
# Input         : None
# Output        : Project title
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def DisplayTitle():

    print("\n===================================================")
    print("          DEEP LEARNING TRAINING PROCESS")
    print("===================================================\n")


# ---------------------------------------------------------
# Function Name : LoadData
# Description   : It initializes input and expected output
# Input         : None
# Output        : Input value and expected output
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def LoadData():

    # STEP 1 : DATA

    x = 2
    actual_output = 10

    print("Input Value     :", x)
    print("Expected Output :", actual_output)

    return x, actual_output


# ---------------------------------------------------------
# Function Name : InitializeSettings
# Description   : It initializes weight and learning rate
# Input         : None
# Output        : Initial weight and learning rate
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def InitializeSettings():

    # STEP 2 : INITIAL SETTINGS

    weight = random.uniform(0, 1)
    learning_rate = 0.1

    print("\nInitial Weight :", weight)
    print("Learning Rate  :", learning_rate)

    return weight, learning_rate


# ---------------------------------------------------------
# Function Name : TrainModel
# Description   : It trains the model using weight update
# Input         : x, actual_output, weight, learning_rate
# Output        : Training values
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def TrainModel(x, actual_output, weight, learning_rate):

    # STEP 3 : STORE VALUES FOR GRAPH

    steps = []
    loss_list = []
    weight_list = []
    prediction_list = []

    # STEP 4 : TRAINING LOOP

    for step in range(1, 21):

        print(f"\n------------ STEP {step} ------------")

        # STEP 5 : FORWARD PASS

        predicted_output = x * weight

        print("Predicted Output :", predicted_output)

        # STEP 6 : ERROR CALCULATION

        error = actual_output - predicted_output

        print("Error            :", error)

        # STEP 7 : LOSS CALCULATION

        loss = error ** 2

        print("Loss             :", loss)

        # STEP 8 : STORE VALUES

        steps.append(step)
        loss_list.append(loss)
        weight_list.append(weight)
        prediction_list.append(predicted_output)

        # STEP 9 : UPDATE WEIGHT

        weight = weight + (learning_rate * error * x)

        print("Updated Weight   :", weight)

    return steps, loss_list, weight_list, prediction_list, weight


# ---------------------------------------------------------
# Function Name : DisplayFinalResult
# Description   : It displays the final prediction result
# Input         : x, actual_output, weight
# Output        : Final training result
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def DisplayFinalResult(x, actual_output, weight):

    # STEP 10 : FINAL RESULT

    print("\n===================================================")
    print("FINAL RESULT")
    print("===================================================\n")

    final_output = x * weight

    print("Final Weight     :", weight)
    print("Final Prediction :", final_output)
    print("Expected Output  :", actual_output)


# ---------------------------------------------------------
# Function Name : PlotLoss
# Description   : It displays loss reduction graph
# Input         : steps, loss_list
# Output        : Loss graph
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def PlotLoss(steps, loss_list):

    # STEP 11 : LOSS REDUCTION GRAPH

    plt.figure()

    plt.plot(steps, loss_list, marker='o')

    plt.title("Loss Decreasing During Training")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid()

    plt.show()


# ---------------------------------------------------------
# Function Name : PlotPrediction
# Description   : It displays prediction versus actual output
# Input         : steps, prediction_list, actual_output
# Output        : Prediction graph
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def PlotPrediction(steps, prediction_list, actual_output):

    # STEP 12 : PREDICTION VS ACTUAL GRAPH

    actual_line = [actual_output] * len(steps)

    plt.figure()

    plt.plot(
        steps,
        prediction_list,
        marker='o',
        label="Predicted Output"
    )

    plt.plot(
        steps,
        actual_line,
        linestyle='--',
        label="Actual Output"
    )

    plt.title("Prediction Approaching Actual Output")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.show()


# ---------------------------------------------------------
# Function Name : PlotWeight
# Description   : It displays weight adjustment graph
# Input         : steps, weight_list
# Output        : Weight change graph
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def PlotWeight(steps, weight_list):

    # STEP 13 : WEIGHT CHANGE GRAPH

    plt.figure()

    plt.plot(steps, weight_list, marker='o')

    plt.title("Weight Adjustment During Learning")
    plt.xlabel("Steps")
    plt.ylabel("Weight Value")
    plt.grid()

    plt.show()


# ---------------------------------------------------------
# Function Name : main
# Description   : It controls the complete training process
# Input         : None
# Output        : Training result and visualization
# Author        : Yash Chavan
# Date          : 11/09/2026
# ---------------------------------------------------------

def main():

    DisplayTitle()

    x, actual_output = LoadData()

    weight, learning_rate = InitializeSettings()

    steps, loss_list, weight_list, prediction_list, weight = TrainModel(
        x,
        actual_output,
        weight,
        learning_rate
    )

    DisplayFinalResult(
        x,
        actual_output,
        weight
    )

    PlotLoss(
        steps,
        loss_list
    )

    PlotPrediction(
        steps,
        prediction_list,
        actual_output
    )

    PlotWeight(
        steps,
        weight_list
    )


if __name__ == "__main__":
    main()