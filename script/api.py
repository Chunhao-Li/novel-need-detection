#-*- coding:utf-8 -*-
import time

import openai
import numpy as np
import pandas as pd
import pickle


df = pd.read_csv('../data/novel_valid_generate_v2_sample.csv')
# df = df.sample(len(df))
sentences_with_label = []
for _,row in df.iterrows():
  sentences_with_label.append((row['sentence1'], row['label']))
print(sentences_with_label[0])
print("length of sentences: ", len(sentences_with_label))

result = []

openai.api_key = ""


prompt = '''Given a computer purchase review, determine whether the review contains “novel needs”. A novel need refers to a function not included in the computer. If there exists novel needs, return 1; if not, or if the comment is meaningless, return 0.
Here are few examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0

You may have misjudged these examples, their labels are 0, not 1:
I like this chromebook, it's easy to hook up and because it only runs on chrome doesn't get too complicated. I am 72 yrs old and not really good with technology so it is a bit of learning curve for me. I would certainly recommend it.Edit: I am dissapointed that I can't do video calls on it. The camera only points to me and not the subject. Didn't realize this until recently.
The MacBook delivers top-notch performance! It’s 14 hours of battery life delivers!!! The bright and colorful display all in is a delightfully portable design. It’s truly worthy of the Air name.
I needed a new laptop for school, but I also wanted something powerful enough for some casual gaming here and there and maybe some video editing down the line. This notebook is able to do both of those! And it's light enough that I don't feel weighed down when it's in my school bag. I'm not a fan of the red keyboard, but it's no deal breaker. I upgraded to a Samsung 1tb nvme SSD and added 8gb more ram for 16 total. This thing is officially a little powerhouse! MSI customer support is also amazing, they answer the phone right away and are very knowledgeable about all of their products. There's tons and tons of information about their products on their website and youtube page. Thanks to that, upgrading this notebook was easy for me (who's otherwise not super knowledgeable about computers). Money well spent!

Given the following review, please return the judgement result:'''

for index,(sent, true_label) in enumerate(sentences_with_label):
  mess = []
  mess.append({"role": "system", "content": "You are a helpful assistant."})
  mess.append({"role": "user", "content": prompt+sent})
  while True:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mess
      )
      break
    except  Exception as e:
      print(f"Encountered an error: {e}. Retrying in 2 seconds...")
      time.sleep(2)


  cur_result = response['choices'][0]['message']['content']
  print(index, cur_result, true_label)
  result.append((index, cur_result, true_label))


print(result)


# with open('../generate_v2_novel_need_result.pkl', 'wb') as f:
#   pickle.dump(result, f)
#
with open('../generate_v2_sample_gpt_result_prompt5.pkl', 'wb') as f:
  pickle.dump(result, f)
