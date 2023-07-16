#-*- coding:utf-8 -*-
import openai
import numpy as np
import pandas as pd
import pickle

# df = pd.read_excel('../data/new_novel_need.xlsx')
# sentences = []
# for _,row in df.iterrows():
#   sentences.append(row['generate_v2'])

df = pd.read_pickle('../data/gpt_infer_neg.pkl')
sentences = []
for _,row in df.iterrows():
  sentences.append(row['neg_text'])
print("length of sentences: ", len(sentences))

result = []

openai.api_key = "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"

# prompt = '''Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements). Here are some examples:
#
# The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.  1
# This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.  1
# The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience.  0
# i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.  0
#
# Given the following reviews, please return the judgement result:'''

prompt = '''Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements). Here are some examples:

The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.  1
This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.  1
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience.  0
i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.  0
I bought this to take my YouTube viewing and general computing on the road. Out of boredom I downloaded a few games yesterday and had quite a pleasant experience! I could actually crank the graphics up most of the way (on an older game, but still!) and kept a steady 60 fps (v-sync'd).I pretty quickly upgraded the RAM and HD. I swapped out the 4GB RAM chip with a 16GB ($88) to give me a total of 20GB (max for this computer). I also upgraded the tiny (but capable) NVMe 128GB hard drive up to 1TB ($109). I downloaded the Windows 10 Home install files onto a USB drive and it went very smoothly. I didn't even need to enter the Windows key, but it might be smart to get your Windows key just incase.Easy mode to do this:- Press Windows key + X.- Right Click Command Prompt (Select Run as Admin)- At the command prompt, type:wmic path SoftwareLicensingService get OA3xOriginalProductKeyThis will reveal the product key. Volume License Product Key Activation.Take a picture of your screen with your phone. But like I said, I didn't need it - it's just in case.The screen isn't anything to write home about but the nuts and bolts of the computer more than make this everything I need when I'm on the road.  0

Given the following reviews, please return the judgement result:'''
# sent = "The cooling system on this computer is top-notch, it runs as cool as a cucumber even under heavy load. But the glossy screen is a nightmare in bright light, it's like looking into a mirror. I long for a matte screen option that eliminates glare and allows me to work anywhere I want."

for index,sent in enumerate(sentences[:100]):
  mess = []
  mess.append({"role": "system", "content": "You are a helpful assistant."})
  mess.append({"role": "user", "content": prompt+sent})

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=mess
  )

  cur_result = response['choices'][0]['message']['content']
  print(index, cur_result)
  result.append((index, cur_result))


print(result)


# with open('../generate_v2_novel_need_result.pkl', 'wb') as f:
#   pickle.dump(result, f)
#
with open('../gpt_infer_neg_100_result.pkl', 'wb') as f:
  pickle.dump(result, f)
