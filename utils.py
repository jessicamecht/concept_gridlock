from torch.nn.utils.rnn import pad_sequence
import clip
scenarios = ["a picture a car in the lane in front which changes the lane", "a picture of an obstacle in the lane in front", 
'a picture of another car in the lane in front that is close by', 'a picture of no car in the lane in front', 'a picture of a truck in the lane in front',
'a picture of a truck in the lane in front which changes the lane', "a picture of cars on a street in rain", "a picture of cars driving on a street in sunny weather", "a picture of cars on a street in cloudy weather", 
'a picture of a street with poor visibility in front', 'a picture of cars driving in the dark', 'a picture of cars driving during the day', "a picture of cars driving on a street with a lot of traffic", 
'a picture of a street with no traffic, there are only few cars', 'a picture of cars driving in a traffic jam',
'a picture of a car which megres into a new street or lane', 'a picture of cars waiting at traffic light',
'a picture of cars driving on a broken road', 'a picture of cars driving close to a construction zone', 'a picture of cars driving and a road sign', 
'a picture of cars driving and pedestrians ahead on the street', 'a picture of cars driving and a bicyle ahead', 
'a picture of a truck on the left', 'a picture of a truck on the right', 'a picture of a car close by on the left', 'a picture of a car close by on the right', 
'a picture of cars driving in the distance', 'a picture of cars driving close by']
scenarios_tokens = clip.tokenize(scenarios)

def pad_collate(batch):
    '''just in case if there were different sequence lengths, 
    but currently all lengths should be the same when batching'''
    meta, img, vego, angle, dist = zip(*batch)
    m_lens = [len(x) for x in meta]
    i_lens = [len(y) for y in img]
    s_lens = [len(x) for x in vego]
    a_lens = [len(y) for y in angle]
    d_lens = [len(y) for y in dist] if dist[0] != None else None 

    m_pad = pad_sequence(meta, batch_first=True, padding_value=0)
    i_pad = pad_sequence(img, batch_first=True, padding_value=0)
    vego_pad = pad_sequence(vego, batch_first=True, padding_value=0)
    a_pad = pad_sequence(angle, batch_first=True, padding_value=0)
    d_pad = pad_sequence(dist, batch_first=True, padding_value=0) if dist[0] != None else None 

    return m_pad, i_pad, vego_pad, a_pad,d_pad, m_lens, i_lens, s_lens, a_lens, d_lens