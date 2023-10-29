import math

import inline as inline
import matplotlib
from geopy.geocoders import Nominatim
from geopy import distance
import numpy as np
import matplotlib.pyplot as plt
# import module
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pywhatkit
import folium
import mysql.connector
import smtplib
from win10toast import ToastNotifier
import time





#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
plt.style.use("seaborn")
np.random.seed(42)


def getdata(url):
	r = requests.get(url)
	return r.text


def find_distance(source,destination):
# initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapiExercises")



# Get location of the input strings
    place1 = geolocator.geocode(source,timeout=None)
    place2 = geolocator.geocode(destination,timeout=None)

    print(place1)
    print(place2)

# Get latitude and longitude
    Loc1_lat, Loc1_lon = (place1.latitude), (place1.longitude)
    Loc2_lat, Loc2_lon = (place2.latitude), (place2.longitude)

    location1 = (Loc1_lat, Loc1_lon)
    location2 = (Loc2_lat, Loc2_lon)

    distance_in_km = distance.distance(location1,location2).km * 1.22228571429
    print(distance_in_km,"Kms")
    return distance_in_km


def draw_map(locations):
    cordinates = list()
    geolocator = Nominatim(user_agent="geoapiExercises")

    for location in locations:
        place = geolocator.geocode(location, timeout=None)
        Loc_lat, Loc_lon = (place.latitude), (place.longitude)
        location_cord = [Loc_lat, Loc_lon]
        cordinates.append(location_cord)

    print(cordinates)

    my_map2 = folium.Map(location=[cordinates[0][0],cordinates[0][1]],
                         zoom_start=10)

    for cordinate in cordinates:
        folium.Marker([cordinate[0], cordinate[1]], popup='Timberline Lodge').add_to(my_map2)

    my_map2.save(" my_map2.html ")

def weather_report(locations):
    report = list()

    for location in locations:
        api_address = 'http://api.openweathermap.org/data/2.5/weather?appid=4d5323b5408d42b832c0f35856aff9f6&q='
        url = api_address + location
        json_data = requests.get(url).json()

        weather = json_data['weather'][0]['main']
        weather_desc = json_data['weather'][0]['description']
        temperature = json_data['main']['temp']
        temperature = float(temperature) - 273.15
        pressure = json_data['main']['pressure']
        humidity = json_data['main']['humidity']
        wind_speed = json_data['wind']['speed']
        name = json_data['name']
        report.append([name,temperature,weather,weather_desc,pressure,humidity,wind_speed])

    weatherdata = pd.DataFrame(report,columns=['CITY NAME','TEMPERATURE IN CELCIUS','WEATHER','DESCRIPTION','PRESSURE IN HECTOPASCAL','HUMIDITY','WIND SPEED IN MPH'])
    return weatherdata




class Population():
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat


    def init_population(self,cities, adjacency_mat, n_population):
        return Population(
            np.asarray([np.random.permutation(cities) for _ in range(n_population)]),
            adjacency_mat
        )

    def fitness(self, chromosome):
        return sum(
            [
                self.adjacency_mat[chromosome[i], chromosome[i + 1]]
                for i in range(len(chromosome) - 1)
            ]
        )

    #Population.fitness = fitness

    def evaluate(self):
        distances = np.asarray(
            [self.fitness(chromosome) for chromosome in self.bag]
        )
        self.score = np.min(distances)
        self.best = self.bag[distances.tolist().index(self.score)]
        self.parents.append(self.best)
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        return distances / np.sum(distances)

    #Population.evaluate = evaluate

    def select(self, k=4):
        fit = self.evaluate()
        while len(self.parents) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.parents.append(self.bag[idx])
        self.parents = np.asarray(self.parents)

    def swap(self,chromosome):
        a, b = np.random.choice(len(chromosome), 2)
        chromosome[a], chromosome[b] = (
            chromosome[b],
            chromosome[a],
        )
        return chromosome

    def crossover(self, p_cross=0.1):
        children = []
        count, size = self.parents.shape
        for _ in range(len(self.bag)):
            if np.random.rand() > p_cross:
                children.append(
                    list(self.parents[np.random.randint(count, size=1)[0]])
                )
            else:
                parent1, parent2 = self.parents[
                                   np.random.randint(count, size=2), :
                                   ]
                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)
                child = [None] * size
                for i in range(start, end + 1, 1):
                    child[i] = parent1[i]
                pointer = 0
                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                children.append(child)
        return children

    def mutate(self, p_cross=0.1, p_mut=0.1):
        next_bag = []
        children = self.crossover(p_cross)
        for child in children:
            if np.random.rand() < p_mut:
                next_bag.append(self.swap(child))
            else:
                next_bag.append(child)
        return next_bag

    def genetic_algorithm(self,
            cities,
            adjacency_mat,
            n_population=5,
            n_iter=20,
            selectivity=0.15,
            p_cross=0.5,
            p_mut=0.1,
            print_interval=100,
            return_history=False,
            verbose=False,
    ):
        pop = self.init_population(cities, adjacency_mat, n_population)
        best = pop.best
        score = float("inf")
        history = []
        for i in range(n_iter):
            pop.select(n_population * selectivity)
            history.append(pop.score)
            if verbose:
                print(f"Generation {i}: {pop.score}")
            elif i % print_interval == 0:
                print(f"Generation {i}: {pop.score}")
            if pop.score < score:
                best = pop.best
                score = pop.score
            children = pop.mutate(p_cross, p_mut)
            pop = Population(children, pop.adjacency_mat)

        if return_history:
            return best, history
        return best


if __name__ == '__main__':
    mileage_of_truck = 5.3
    cities = list()
    while (True):
        city = input("Enter the city name and type 'exit' to end\n")
        if city == "exit":
            break
        cities.append(city)
    print(cities)

    cities_list = list()
    for i in range(len(cities)):
        cities_list.append(i)


    distance_matrix = [[find_distance(source,destination) for source in cities] for destination in cities]
    distance_matrix = np.asarray(distance_matrix)
    print(distance_matrix)

    #route = Population(cities_list,distance_matrix)
    #route = route.init_population(cities_list,distance_matrix,len(cities))
    #print(route.bag)
    #pop = route.evaluate()
    #print(pop)
    #print(route.best)
    #print(route.score)
    #route.select()
    #print(route.parents)
    #pop = route.mutate()
    #print(pop)

    #route1 = Population(cities_list,distance_matrix)
    #route1 = route1.genetic_algorithm(cities_list,distance_matrix,verbose=True)
    #print(route1)

    route2 = Population(cities_list,distance_matrix)
    best,history = route2.genetic_algorithm(cities_list,distance_matrix,n_iter=100,verbose=False,print_interval=10,return_history=True)
    print(best)
    #print(history)
    plt.plot(range(len(history)), history, color="skyblue")
    plt.show()
    print("best route")
    ord = list()
    for elem in best:
        ord.append(cities[elem])
        print(cities[elem])

    best_distance = history[-1]
    print("Best distance in kms",round(best_distance,3),"Kms")

    # link for extract html data
    htmldata = getdata("https://www.goodreturns.in/diesel-price.html")
    soup = BeautifulSoup(htmldata, 'html.parser')
    #result = soup.find_all("div", class_="gold_silver_table")
    #print(result)

    # Declare string var
    # Declare list
    mydatastr = ''
    result = []

    # searching all tr in the html data
    # storing as a string
    for table in soup.find_all('tr'):
        mydatastr += table.get_text()

    # set accourding to your required
    mydatastr = mydatastr[1:]
    itemlist = mydatastr.split("\n\n")

    for item in itemlist[:-5]:
        result.append(item.split("\n"))

    #print(result)

    location = list()
    for i in result:
        location.append(i[0].lower())
    print(location)
    ind = list()
    for elem in cities:
        #print("Location is ", elem)
        # print('chennai' in location)
        if elem in location:
            #print('found')
            ind.append(location.index(elem))
        #print(ind)

    # Calling DataFrame constructor on list
    df = pd.DataFrame(result[:-8])
    print(df)

    price_per_day = list()
    price = list()
    cities_present = list()
    for elem in ind:
        cities_present.append(elem)
        fuel_price = result[elem][1]
        fuel_price = float(fuel_price[2:])
        price_per_day.append(fuel_price)
        # print("Today's petrol diesel in rupess ", fuel_price)
        cost_for_fuel = (fuel_price * best_distance) / mileage_of_truck
        price.append(cost_for_fuel)
        # print("Total cost for fuel in rupess ", round(cost_for_fuel, 2))

    #print(cities_present)
    i = 0
    for elem in cities_present:
        cities_present[i] = location[int(elem)].capitalize()
        i += 1

    info = {
        "Price per day": price_per_day,
        "Total price": price
    }
    print("\n")
    df1 = pd.DataFrame(info, index=cities_present)
    print(df1)
    print("\n")



    city_code = int(input("Enter the city code for filling diesel\n"))
    fuel_price = result[city_code][1]
    fuel_price = float(fuel_price[2:])
    print("Today's petrol diesel in rupess ", fuel_price)
    cost_for_fuel = (fuel_price * best_distance) / mileage_of_truck
    print("Total cost for fuel in rupess ",round(cost_for_fuel,2))
    average_speed = float(input("Enter your expected average speed\n"))

    time_list = list()
    temp = average_speed - 10
    for i in range(5):
        minute = (best_distance / temp)
        minute = (minute/1) - (minute//1)
        hour = int(best_distance // temp)
        print("Time in hours to cover", round(best_distance,3), "Kms with", int(temp), "KMPH as average", hour, "hours and",
              int(minute * 60), "minutes")
        time_list.append([int(temp),hour,int(minute * 60)])
        temp += 2

    temp = average_speed
    for i in range(6):
        minute = (best_distance/temp)
        minute = (minute/1) - (minute//1)
        hour = int(best_distance // temp)
        print("Time in hours to cover",round(best_distance,3),"Kms with",int(temp), "KMPH as average",hour,"hours and",int(minute*60),"minutes")
        time_list.append([int(temp),hour,int(minute * 60)])
        temp += 2

    draw_map(cities)
    weatherdata = weather_report(cities)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    dataframe = pd.DataFrame(time_list,columns=['SPEED IN KMPH','HOURS','MINUTES'])
    order = ' -> '.join(map(str,ord))

    print("+===================================================================================+")
    print("| ENTERED CITIES                    :   ",cities)
    print("| OPTIMUM ORDER FOR TRAVEL          :   ",order)
    print("| OPTIMAL DISTANCE IN KMS           :   ",best_distance)
    print("| FUEL PRICE PER LITER IN RUPEES    :   ",fuel_price)
    print("| TOTAL COST FOR FUELING IN RUPEES  :   ",cost_for_fuel)
    print("| TIME TO TRAVEL",best_distance,"KMS")
    print(dataframe)
    print(weatherdata)
    print("+===================================================================================+")

    best_distance = str(best_distance)
    fuel_price = str(fuel_price)
    cost_for_fuel = str(cost_for_fuel)
    dataframe = str(dataframe)

    conn = mysql.connector.connect(host="localhost", port="3306", user="sabhari", password="2000",
                                   database="vehicle_routing")
    cursorpy = conn.cursor()

    query = "INSERT INTO `trip`( `best_route`, `best_distance`, `fuel_per_liter`, `cost_for_fuel`) VALUES (%s,%s,%s,%s)"
    values = (order,best_distance,fuel_price,cost_for_fuel)
    cursorpy.execute(query, values)
    conn.commit()
    cursorpy.close()
    conn.close()

    list_of_branch = list()
    list_of_branch.append("ajaysengottuvel1610@gmail.com")
    list_of_branch.append("abhiram234k@gmail.com")

    for mail_id in list_of_branch:

        s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
        s.starttls()

    # Authentication
        s.login("@gmail.com", "password")

        order = '\n'.join(map(str, cities))

    # message to be sent
        message = "SUB : About the planned trip\nHereby we had planned a trip to \n"+order+"\nWith a total distance of "+best_distance+"KMS\nWith the total fuel price of "+ cost_for_fuel + "\nAverage time as\n"+dataframe+".\n"

    # sending the mail
        s.sendmail("sabhari2000@gmail.com",mail_id, message)

    # terminating the session
        s.quit()

    custom_notification = ToastNotifier()
    custom_notification.show_toast("COMMUNICATION SUCCESFULL NOTIFICATION",
                                   "THE LOCATION INFORMATIONS HAD BEEN SHARED WITH BRANCHES THROUGH MAIL",
                                   threaded=True, icon_path=None, duration=10)
    while custom_notification.notification_active():
        time.sleep(0.1)

    hour = int(input("Enter hour for messaging"))
    minute = int(input("Enter minute for messaging"))

    pywhatkit.sendwhatmsg('+919842824705','Hi ajay! this message is from company to inform you that a trip had been booked for you. kindly refer your portal for full information',hour,minute)
    #pywhatkit.sendwhatmsg_to_group('CZBRNqeHD9K6HruYK6aOBR','Hi guys! this message is from company to inform you that a trip had been booked for you. kindly refer your portal for full information',hour,minute)
