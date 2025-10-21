import pandas as pd 


features = pd.read_csv("audio_features_aggregated.csv")
daily_fi = pd.read_excel("daily_fi.xlsx")
date_convert = pd.read_excel("date_convert.xlsx")
date_convert["dt"] = date_convert["Date"].apply(lambda x:x.date())
date_convert["hour"] = date_convert["Date"].apply(lambda x:x.hour)



features["date"] = features["file"].apply(lambda x:x.split("_")[1])
features["noise_reduction"] = features["file"].apply(lambda x:x.split("_")[2])
features["hour"] = features["file"].apply(lambda x:x.split("_")[3].split(".")[0].split("-")[0])
features["pen"] = features["file"].apply(lambda x:x.split("_")[0].replace("rec",""))



"""for i in range(len(features)):
    dt = datetime.datetime.strptime(features["date"][i]+" "+features["hour"][i],"%Y-%m-%d %H")
    date_title = date_convert[date_convert["date"]]["title"].values[0]

features = features.merge(daily_fi,on=["pen","date","hour"])

"""

features.to_excel("feartures.xlsx")

features_nr = features[features["noise_reduction"]=="NR"]
features_nonr = features[features["noise_reduction"]=="noNR"]

