import pandas as pd

# Define the data
data = {
    "Metric": ["Coefficient of Variation", "Entropy", "Normalized Entropy", "Functional Complexity"],
    "Advantage": [
        "1. Simple and intuitive measure of spike train variability.\n2. Gives a standardized measure of dispersion.\n3. Allows for easy interpretation.\n4. Less sensitive to large values.",
        "1. Provides a measure of the unpredictability or randomness of the spike train.\n2. Can capture the temporal structure of the spike train.\n3. Not dependent on the mean or standard deviation.\n4. Can provide insights into the informational content of spike trains.",
        "1. Provides a relative measure of unpredictability or randomness of the spike train, adjusted for the maximum possible entropy.\n2. Still captures the temporal structure of the spike train.\n3. Can give a standardized metric.\n4. Provides insights into the informational content of spike trains relative to their maximum potential information content.",
        "1. Provides insights into the range and diversity of a neuron's functional properties.\n2. Can account for a wide variety of neuronal behaviors.\n3. Can be customized to reflect specific aspects of functional complexity (e.g., receptor diversity, synaptic plasticity).\n4. Complements other metrics by providing a more holistic view of neuronal function."
    ],
    "Disadvantage": [
        "1. Does not provide any information on the temporal structure of the spike train.\n2. Assumes a unimodal and symmetrical distribution.\n3. Can be greatly affected by changes in mean spike count.",
        "1. Calculation of entropy can be complex and computationally expensive.\n2. Requires a large number of spike trains for reliable estimates.\n3. Might require additional interpretation.\n4. Might not be as intuitive to understand or interpret as CV.",
        "1. Still requires a complex and potentially computationally expensive calculation.\n2. Requires a large number of spike trains for reliable estimates.\n3. The interpretation might still be challenging compared to more straightforward measures like CV.\n4. Determining the maximum possible entropy for a given time window and rate can be difficult and might introduce additional sources of error or uncertainty.",
        "1. Requires detailed knowledge of the neuron's functional properties, which might not always be available or easily measured.\n2. Assessing functional complexity can be technically challenging and may require specialized experimental techniques.\n3. Interpretation can be complex, as functional properties are influenced by a wide range of factors and can change dynamically in response to different conditions.\n4. No single metric can fully capture functional complexity, requiring a multifaceted approach."
    ]
}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

# Save the DataFrame into an Excel file
df.to_excel("table.xlsx", index=False)
