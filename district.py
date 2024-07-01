# Dictionary to store district data
district_data = {}

# Function to add a new district entry
def add_district(name, population, area, other_attributes={}):
    if name in district_data:
        print(f"District {name} already exists.")
        return
    district_data[name] = {
        'population': population,
        'area': area,
        **other_attributes
    }
    print(f"District {name} added successfully.")

# Function to update an existing district entry
def update_district(name, population=None, area=None, other_attributes={}):
    if name not in district_data:
        print(f"District {name} does not exist.")
        return
    if population is not None:
        district_data[name]['population'] = population
    if area is not None:
        district_data[name]['area'] = area
    district_data[name].update(other_attributes)
    print(f"District {name} updated successfully.")

# Function to display district data
def display_district(name):
    if name not in district_data:
        print(f"District {name} does not exist.")
        return
    print(f"District {name}:")
    for key, value in district_data[name].items():
        print(f"  {key}: {value}")

# Example usage
add_district("Alwar", 500000, 1200, )
add_district("Bhratpur", 750000, 1500,)

display_district("Alwar")

update_district("Alwar", population=550000, )

display_district("bhratpur")