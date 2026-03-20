import random
import csv

def generate_plant_data(num_plants=1000):
    """Generates mock plant data with maintenance schedules and fertilizer details."""

    plant_types = [
        "Rose", "Tulip", "Sunflower", "Daisy", "Fern", "Cactus", "Orchid",
        "Peace Lily", "Snake Plant", "ZZ Plant", "Monstera", "Pothos",
        "Fiddle Leaf Fig", "Aloe Vera", "Lavender", "Mint", "Basil", "Tomato",
        "Cucumber", "Pepper"  # Add more plant types as needed
    ]

    fertilizers = {
        "Balanced": "10-10-10",  # NPK ratio
        "High Nitrogen": "20-5-5",
        "High Phosphorus": "5-15-5",
        "High Potassium": "5-5-15",
        "Organic": "Compost or Worm Castings"  # No specific NPK for organic
    }

    watering_frequencies = ["Daily", "Every other day", "Twice a week", "Weekly"]
    sunlight_needs = ["Full sun", "Partial shade", "Full shade"]

    plant_data = []

    for _ in range(num_plants):
        plant_type = random.choice(plant_types)
        fertilizer_type = random.choice(list(fertilizers.keys()))
        fertilizer_quantity = random.choice(["1/4 tsp", "1/2 tsp", "1 tsp", "1 tbsp"])  # Adjust quantities

        watering_frequency = random.choice(watering_frequencies)
        sunlight_need = random.choice(sunlight_needs)
        pruning = random.choice(["Monthly", "Quarterly", "Annually", "As needed"])


        plant_data.append({
            "Plant Type": plant_type,
            "Fertilizer Type": fertilizer_type,
            "Fertilizer Quantity": fertilizer_quantity,
            "Fertilizer NPK": fertilizers[fertilizer_type] if fertilizer_type != "Organic" else "N/A",
            "Watering Frequency": watering_frequency,
            "Sunlight Needs": sunlight_need,
            "Pruning": pruning
        })

    return plant_data

def save_to_csv(plant_data, filename="plant_data.csv"):
    """Saves plant data to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = plant_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(plant_data)




# Generate the data
plant_data = generate_plant_data()


# Save to CSV
save_to_csv(plant_data)

print(f"Generated data for {len(plant_data)} plants and saved to plant_data.csv")



import csv
import random

def get_plant_recommendations(sick_plant_name, plant_data_file="plant_data.csv"):
    """
    Matches a sick plant with data from the plant_data file and provides care recommendations.
    """
    try:
        with open(plant_data_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            matching_plants = [row for row in reader if row["Plant Type"] == sick_plant_name]

    except FileNotFoundError:
        return f"Error: Plant data file '{plant_data_file}' not found."


    if matching_plants:
        chosen_plant = random.choice(matching_plants)  # Choose a random match if there are multiple

        recommendations = (
            f"Care recommendations for your {sick_plant_name}:\n"
            f"- Fertilizer: {chosen_plant['Fertilizer Type']} ({chosen_plant['Fertilizer Quantity']})\n"  # Include quantity
            f"- Watering: {chosen_plant['Watering Frequency']}\n"
            f"- Sunlight: {chosen_plant['Sunlight Needs']}\n"
            f"- Pruning: {chosen_plant['Pruning']}"
        )
        return recommendations

    else:  # No matching plant was found. Try partial matching or fuzzy matching
        possible_matches = []
        with open(plant_data_file, 'r', newline='') as csvfile:
             reader = csv.DictReader(csvfile)
             for row in reader:
                if sick_plant_name.lower() in row["Plant Type"].lower(): # Case-insensitive partial match
                     possible_matches.append(row["Plant Type"])

        if possible_matches: # Offer suggestions based on the matches found
            suggestions = f"No exact match for '{sick_plant_name}' found. Did you mean: {', '.join(possible_matches)}?"
            return suggestions
        else: # Nothing similar was found at all
            return f"No information found for '{sick_plant_name}'. Check the spelling or try a different plant."





# Example usage:
sick_plant = "Rose"  # Replace with the name of the sick plant
recommendations = get_plant_recommendations(sick_plant)
print(recommendations)


sick_plant = "rose"  # Example with different capitalization
recommendations = get_plant_recommendations(sick_plant)
print(recommendations)


sick_plant = "Tulip" # Test with a plant that's definitely present
recommendations = get_plant_recommendations(sick_plant)
print(recommendations)

sick_plant = "Ro"  # Example with partial match (will offer suggestions). Test your fuzzy/partial matching
recommendations = get_plant_recommendations(sick_plant)
print(recommendations)

sick_plant = "NonExistentPlant" # Test with something completely different
recommendations = get_plant_recommendations(sick_plant)
print(recommendations)




