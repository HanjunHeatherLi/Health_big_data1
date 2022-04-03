import pandas as pd
import numpy as np
import utils

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''

    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    dead_list = mortality["patient_id"].unique()
    alive = events[~events["patient_id"].isin(dead_list)]
    alive_indx = alive.groupby("patient_id").timestamp.max()
    alive_indxdf=alive_indx.to_frame().reset_index()

    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'])
    mortality['dead_indx']=mortality['timestamp']-pd.Timedelta(days=30)
    dead_indxdf=mortality.drop(['timestamp', 'label'], axis=1).rename(columns={"dead_indx":"timestamp"})
    indx_datedf=pd.concat([alive_indxdf,dead_indxdf]).reset_index()
    indx_date=indx_datedf.rename(columns={"timestamp":"indx_date"})
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''


    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    mergedf = pd.merge(events, indx_date, on="patient_id")
    mergedf['timestamp'] = pd.to_datetime(mergedf['timestamp'])
    selected_df = mergedf.loc[((mergedf['indx_date']-mergedf['timestamp']).dt.days <= 2000) & ((mergedf['indx_date']-mergedf['timestamp']).dt.days >= 0) ]

    filtered_events = selected_df[["patient_id","event_id","value"]]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'],
                           index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''


    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''


    idxed_event = pd.merge(filtered_events_df,feature_map_df, how='left', on='event_id').drop(["event_id"], axis=1).dropna()

    feature_value = idxed_event.groupby(["patient_id","idx"]).count().reset_index()
    #print(feature_value)
    feature_value_max = feature_value.drop(["patient_id"],axis=1).groupby("idx").agg({'value': [np.max]}).reset_index()
    #print(feature_value_max)
    aggregated_events = pd.merge(feature_value, feature_value_max, how='left', on='idx')
    #print(aggregated_events)
    aggregated_events["feature_value"] = (aggregated_events["value"]) / (aggregated_events[("value","amax")])
    aggregated_events = aggregated_events.rename(columns={"idx": "feature_id"})
    aggregated_events = aggregated_events[['patient_id', 'feature_id', 'feature_value']]
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''


    #mortality = {}


    pd_mortality = pd.merge(aggregated_events, mortality, how='left', on="patient_id").fillna(0).iloc[:, [0, 4]].set_index('patient_id').T.to_dict("list")
    #print(pd_mortality)

    aggregated_events["total"] = aggregated_events.apply(lambda row: [int(row["feature_id"]), row["feature_value"]],axis=1)
    #print(aggregated_events)
    aggregated_events.drop(["feature_id", "feature_value"], axis=1, inplace=True)
    aggregated_events = aggregated_events.groupby('patient_id')['total'].apply(list).to_frame()
    patient_features = {}
    for index, row in aggregated_events.iterrows():
        patient_features[index] = row["total"]

    return patient_features, {x: y[0] for x, y in pd_mortality.items()}

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    #sort patients by id, sort feature in ascending order.
    sorted_patient_features=sorted(patient_features.items())
    #print(sorted_patient_features[0])
    for key, item in sorted_patient_features:
        item.sort()

        deliverable1.write(bytes(str(int(mortality[key])) + " " + " ".join(
            ":".join([str(int(i[0])), str(format(i[1], '.6f'))]) for i in item)
                                 + " " + "\n", 'UTF-8'))

        deliverable2.write(bytes(" ".join([str(int(key)), str(int(mortality[key]))]) + " " +
                                 " ".join(":".join([str(int(i[0])), str(format(i[1], '.6f'))]) for i in item)
                                 + " " + "\n", 'UTF-8'))



def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()