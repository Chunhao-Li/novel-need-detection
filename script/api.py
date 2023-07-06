#-*- coding:utf-8 -*-
import os
import openai

openai.api_key = "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"

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
mess = []
prompt = '''Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements). Here are some examples:

The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.  1 
This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.  1 
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience.  0 
i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.  0

Given the following reviews, please return the judgement result:'''
sent = "The cooling system on this computer is top-notch, it runs as cool as a cucumber even under heavy load. But the glossy screen is a nightmare in bright light, it's like looking into a mirror. I long for a matte screen option that eliminates glare and allows me to work anywhere I want."

mess.append({"role": "system", "content": "You are a helpful assistant."})
mess.append({"role": "user", "content": prompt+sent})

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=mess
)

print(response)
# print("\u8fd9\u91cc\u6ca1\u6709\u63d0\u4f9b\u6b63\u786e\u7684\u9009\u9879\uff0c\u9700\u8981\u63d0\u4f9b\u6b63\u786e\u9009\u9879\u624d\u80fd\u5224\u65ad\u54ea\u4e2a\u662f\u6b63\u786e\u7684\u3002")
print(response['choices'][0]['message']['content'])
# print(response['choices'][0]['text'])