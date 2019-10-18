import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import urllib,urllib2,json,re,datetime,sys,cookielib

import codecs
from pyquery import PyQuery
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys

from datetime import timedelta,datetime
from dateutil import parser

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from stop_words import safe_get_stop_words
from stop_words import get_stop_words
from textblob import TextBlob

from BeautifulSoup import BeautifulSoup 
import numpy as np
import random
import xlsxwriter
import time
import math
import numpy as np
import glob
import os
import sys
reload(sys)

sys.setdefaultencoding('utf8')

class Tweet:
	
	def __init__(self):
		pass


class TweetCriteria:
	
	def __init__(self):
		self.maxTweets = 0
		
	def setUsername(self, username):
		self.username = username
		return self
		
	def setSince(self, since):
		self.since = since
		return self
	
	def setUntil(self, until):
		self.until = until
		return self
		
	def setQuerySearch(self, querySearch):
		self.querySearch = querySearch
		return self
		
	def setMaxTweets(self, maxTweets):
		self.maxTweets = maxTweets
		return self

	def setTopTweets(self, topTweets):
		self.topTweets = topTweets
		return self

####code starts from here


class TweetManager:
	
	def __init__(self):
		pass
		
	@staticmethod
	def getTweets(tweetCriteria, receiveBuffer , bufferLength = 100):
		refreshCursor = ''
	        #receiveBuffer=receiveBuffer1
		results = []
		resultsAux = []
		cookieJar = cookielib.CookieJar()
		
		if hasattr(tweetCriteria, 'username') and (tweetCriteria.username.startswith("\'") or tweetCriteria.username.startswith("\"")) and (tweetCriteria.username.endswith("\'") or tweetCriteria.username.endswith("\"")):
			tweetCriteria.username = tweetCriteria.username[1:-1]

		active = True

		while active:
		        json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar)
			if len(json['items_html'].strip()) == 0:
				break

			refreshCursor = json['min_position']			
			tweets = PyQuery(json['items_html'])('div.js-stream-tweet')
			
			if len(tweets) == 0:
				break
			
			for tweetHTML in tweets:
				tweetPQ = PyQuery(tweetHTML)
				tweet = Tweet()
				
				usernameTweet = tweetPQ("span.username.js-action-profile-name b").text();
				txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'));
				retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
				id = tweetPQ.attr("data-tweet-id");
				permalink = tweetPQ.attr("data-permalink-path");
				
				geo = ''
				geoSpan = tweetPQ('span.Tweet-geo')
				if len(geoSpan) > 0:
					geo = geoSpan.attr('title')
				
				tweet.id = id
				tweet.permalink = 'https://twitter.com' + permalink
				tweet.username = usernameTweet
				tweet.text = txt
				tweet.date = datetime.fromtimestamp(dateSec)
				tweet.retweets = retweets
				tweet.favorites = favorites
				tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
				tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
				tweet.geo = geo
				
				results.append(tweet)
				resultsAux.append(tweet)
				
				if receiveBuffer and len(resultsAux) >= bufferLength:
					receiveBuffer(resultsAux)
					resultsAux = []
				
				if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
					active = False
					break
					
		
		if receiveBuffer and len(resultsAux) > 0:
			receiveBuffer(resultsAux)
		
		return results
	
	@staticmethod
        def getJsonReponse(tweetCriteria, refreshCursor, cookieJar):
		url = "https://twitter.com/i/search/timeline?f=tweets&q=%s&src=typd&max_position=%s"
		
		urlGetData = ''
		if hasattr(tweetCriteria, 'username'):
			urlGetData += ' from:' + tweetCriteria.username
			
		if hasattr(tweetCriteria, 'since'):
			urlGetData += ' since:' + tweetCriteria.since
			
		if hasattr(tweetCriteria, 'until'):
			urlGetData += ' until:' + tweetCriteria.until
			
		if hasattr(tweetCriteria, 'querySearch'):
			urlGetData += ' ' + tweetCriteria.querySearch

		if hasattr(tweetCriteria, 'topTweets'):
			if tweetCriteria.topTweets:
				url = "https://twitter.com/i/search/timeline?q=%s&src=typd&max_position=%s"

		url = url % (urllib.quote(urlGetData), refreshCursor)

		headers = [
			('Host', "twitter.com"),
			('User-Agent', "Mozilla/4.0 (Windows NT 6.1; Win64; x64)"),
			('Accept', "application/json, text/javascript, */*; q=0.01"),
			('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
			('X-Requested-With', "XMLHttpRequest"),
			('Referer', url),
			('Connection', "keep-alive")
		]

		opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookieJar))
		opener.addheaders = headers

		try:
			response = opener.open(url)
			jsonResponse = response.read()
		except:
		        #pass  
			print "Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.quote(urlGetData)			
			sys.exit()
			return
		
		dataJson = json.loads(jsonResponse)
		
		return dataJson		
def receiveBuffer1(tweets):
 for t in tweets:
  outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites,  t.geo, t.mentions, t.hashtags, t.id, t.permalink,t.text)))
  outputFile.flush()
		
				
def get_senti_score(all_tweets_df):
		sia = SentimentIntensityAnalyzer()
		regex_str = "http[s]?:// (?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+|(?:@[\w_]+)|(?:\#+[\w_]+[\w\'_\-]*[\w_]+)|http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+|(?:(?:\d+,?)+(?:\.?\d+)?)|(-)"
		stop_words1 = list(set(stopwords.words('english')))
		stop_words1 = " ".join(i.encode('utf-8') for i in stop_words1).split()
		stop_words2 = []
		stop_words3 = list(get_stop_words('en'))
		stop_words3 = " ".join(i.encode('utf-8') for i in stop_words3).split()
		stop_words4 = list(safe_get_stop_words('unsupported language'))
		stop_words = list(pd.unique(stop_words1+stop_words2+stop_words3+stop_words4))
		
		score = []
		for j in range(0,len(all_tweets_df)): 
			x = re.sub(regex_str,"",str(all_tweets_df.text[j]))
			x = x.lower()
			x = unicode(str(x), 'utf-8')
			x = TextBlob(x)
			stop_words6 = list(x.noun_phrases)
			y = " ".join(p.encode('utf-8') for p in stop_words6).split()
			stop_words_new = stop_words + y
			x = " ".join(p for p in x.split() if p not in stop_words_new)
			
			score.append(sia.polarity_scores(x)['compound'])
		
		value = round(sum(score)/len(all_tweets_df),5)
		return value
		
def get_tweets_text(df_excel,CompanyURLs,path):
                #path = 'A:/Capstone_Code/twitter/'
		f = lambda x: x.rpartition("/")[len(x.rpartition("/"))-1]
		i=1
		filelist_del = glob.glob(os.path.join(path, "*.csv"))
                for f_del in filelist_del:
                   os.remove(f_del)
		for i in range(1,len(CompanyURLs)+1):
			if pd.notnull(CompanyURLs.twitterLink[i]):
				StartDate = (datetime.now() - timedelta(365)).strftime('%Y-%m-%d')
				EndDate = datetime.now().strftime('%Y-%m-%d')
				twitter_ticker = f(CompanyURLs.twitterLink[i])				
				tweetCriteria = TweetCriteria().setQuerySearch('@'+twitter_ticker).setSince(StartDate).setUntil(EndDate)
				outputFile = codecs.open(path+str(i)+"-"+'comp1'+str(i)+".csv", "w+", "utf-8")		
                                outputFile.write('username;date;retweets;favorites;geo;mentions;hashtags;id;permalink;text') 
                                TweetManager.getTweets(tweetCriteria, receiveBuffer1)
				all_tweets_df = glob.glob(os.path.join(path, "*.csv"))
				senti_score = get_senti_score(all_tweets_df)
				df_excel.at[df_excel['Target Company'] == CompanyURLs.company[i],'sentiment_score'] = senti_score
				df_excel.at[df_excel['Target Company'] == CompanyURLs.company[i],'No_of_tweets'] = len(all_tweets_df)
				df_excel.at[df_excel['Target Company'] == CompanyURLs.company[i],'retweets'] = sum(all_tweets_df.retweets)
				df_excel.at[df_excel['Target Company'] == CompanyURLs.company[i],'favourites'] = sum(all_tweets_df.favorites)
				
		return df_excel
			
	

	
def get_twitter_url(df_excel,CompanyURLs):
		#
		df_excel.reset_index()
		df_excel.index = np.arange(1, len(df_excel) + 1)
		browser1=urllib2.build_opener()
                browser1.addheaders=[('User-agent', 'Mozilla/5.0')]

		for i in range(1,len(df_excel)+1):
		   # i=2
			if not pd.isnull(df_excel.URL[i]):
				try:		        
					resp =browser1.open(df_excel.URL[i])
					html = resp.read()
				        time.sleep(18) 
					soup = BeautifulSoup(html)					
					CompanyURLs.at[i,'company'] = df_excel.loc[i,'Target Company']
					CompanyURLs.at[i,'website'] = df_excel.URL[i]+"aa"
					p=0
					for link in soup.findAll("a"):
					        link = soup.findAll("a")[p]	
						if 'twitter' in str(link) and "twitter" in link['href']:
							 CompanyURLs.at[i,'twitterLink'] = link['href']   
						p=p+1	    
				except:
				        
					CompanyURLs.at[i,'company'] = df_excel.loc[i,'Target Company']
					CompanyURLs.at[i,'website'] = df_excel.loc[i,'URL']
					CompanyURLs.at[i,'twitterLink'] = ''
			else:
			        #print i,"else"
				CompanyURLs.at[i,'company'] =df_excel.loc[i,'Target Company']
				CompanyURLs.at[i,'website'] = df_excel.loc[i,'URL']
				CompanyURLs.at[i,'twitterLink'] = ''
		
	        return CompanyURLs   #return df_excel,CompanyURLs
 

    
def getpeopleinfo(bsObj1,company_name,UnqId):
    people_df_mod = pd.DataFrame( columns = ['Person_Name','Title','UnqId','Age','Number_Relations','Company_Name','Person_Id','URL','Primary_Company'])
    table_exec = bsObj1.find("table", attrs={ "id" : "keyExecs" })
    table_board = bsObj1.findAll("table", attrs={ "class" : "table" })
    i_ppl_indx = 1
    reset_indx = 1
    if str(table_exec) <> 'None':
     for row in table_exec.findAll("tr"):         
         cells = row.findAll("td")
         if reset_indx>1 and len(cells) >1 :
           people_df_mod.at[i_ppl_indx,'Company_Name'] = company_name
           people_df_mod.at[i_ppl_indx,'UnqId'] = UnqId
           people_df_mod.at[i_ppl_indx,'Person_Id'] = i_ppl_indx
           if 'No Relationships' not in cells[1].text:
            people_df_mod.at[i_ppl_indx,'Number_Relations'] = cells[1].find("strong").text
            people_df_mod.at[i_ppl_indx,'URL'] = cells[1].find("a")['href'].replace("../..","https://www.bloomberg.com/research")
           people_df_mod.at[i_ppl_indx,'Person_Name'] = cells[0].text
           people_df_mod.at[i_ppl_indx,'Title'] = cells[2].text
           people_df_mod.at[i_ppl_indx,'Age'] = cells[3].text           
           i_ppl_indx = i_ppl_indx+1
         reset_indx =reset_indx+1
    reset_indx = 1 
    if str(table_board) <> 'None' and len(table_board)>0:    
     for row in table_board[1].findAll("tr"):         
         cells = row.findAll("td")
         if reset_indx>1 and len(cells) >1 :
           people_df_mod.at[i_ppl_indx,'Company_Name'] = company_name
           people_df_mod.at[i_ppl_indx,'UnqId'] = UnqId
           people_df_mod.at[i_ppl_indx,'Person_Id'] = i_ppl_indx
           if 'No Relationships' not in cells[1].text:
            people_df_mod.at[i_ppl_indx,'Number_Relations'] = cells[1].find("strong").text
            people_df_mod.at[i_ppl_indx,'URL'] = cells[1].find("a")['href'].replace("../..","https://www.bloomberg.com/research")
           people_df_mod.at[i_ppl_indx,'Person_Name'] = cells[0].text
           people_df_mod.at[i_ppl_indx,'Primary_Company'] = cells[2].text
           people_df_mod.at[i_ppl_indx,'Age'] = cells[3].text  
           people_df_mod.at[i_ppl_indx,'Title']  = 'Board Member'       
           i_ppl_indx = i_ppl_indx+1
         reset_indx =reset_indx+1

    return people_df_mod

def getrelations(people_df,browser1):
    relation_df_mod = pd.DataFrame( columns = ['Person_Id','Person_Name','UnqId','Relation_Name','Relation_Company'])
    rel_indx_all = 1
    relation_list=[]
    for i_rel_indx in xrange(1,len(people_df)+1):
     if str(people_df.loc[i_rel_indx,'URL']) <> 'nan':
         try:
            response2=browser1.open(people_df.loc[i_rel_indx,'URL'],timeout=(10.0))
            sleep_time1 = random.randint(2,20)
            time.sleep(sleep_time1) 
            html2 = response2.read()
            bsObj2 = BeautifulSoup(html2)  
            relation_list = bsObj2.findAll("div",attrs={"class":"relationBox"})
         except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            print 'ERROR FOR URL:',people_df.loc[i_rel_indx,'URL']
            print error_type, 'Line:', error_info.tb_lineno
            continue
#Initializing Beautifulsoup object

     #relations = relation_list[0]
         for relations in relation_list:   
            relation_df_mod.at[rel_indx_all,'Person_Id'] = people_df.loc[i_rel_indx,'Person_Id']
            relation_df_mod.at[rel_indx_all,'Person_Name'] = people_df.loc[i_rel_indx,'Person_Name']
            relation_df_mod.at[rel_indx_all,'UnqId'] = people_df.loc[i_rel_indx,'UnqId']
            relation_df_mod.at[rel_indx_all,'Relation_Name'] = relations.findAll("a")[0].text 
            relation_df_mod.at[rel_indx_all,'Relation_Company'] = relations.findAll("a")[1].text 
            rel_indx_all=rel_indx_all+1
    
    return relation_df_mod
    
def getCompanyAttributes(data_pd_final):
    df_company_attribs = pd.DataFrame()
    df_comp_buyer =pd.DataFrame()
    df_company_attribs['Company']=data_pd_final['Target Company']
    df_company_attribs['Type']='Target'    
    df_comp_buyer['Company']=data_pd_final['Buyer']
    df_comp_buyer['Type']='Buyer'
    df_company_attribs = df_company_attribs.append(df_comp_buyer)
    df_company_attribs.reset_index()
    df_company_attribs.index = np.arange(1, len(df_company_attribs) + 1)    
    
    
    url_df = pd.DataFrame( columns = ['URL', 'Company','UnqId'])
    url_df_final = pd.DataFrame( columns = ['URL', 'Company','UnqId'])
    url_df_mod =  pd.DataFrame( columns = ['URL', 'Company','UnqId'])
    final_url_df = pd.DataFrame(columns = ['URL', 'Company','UnqId','Type'])
    data_pd = pd.DataFrame( columns = ['Company','Type','Missing'])

    browser = webdriver.Firefox(executable_path='A:\geckodriver-v0.19.0-win64\geckodriver.exe')
    time.sleep(5)
    browser.get('http://www.yahoo.com')
    comp_names_indx=1
    p_indx_url_exists = 1
    for comp_names_indx in xrange(1,len(df_company_attribs)+1): 
      sleep_time = random.randint(1,10)
      company_name = df_company_attribs.loc[comp_names_indx,'Company']
      comp_type = df_company_attribs.loc[comp_names_indx,'Type']    
      try:
        wait = WebDriverWait(browser, 30)
        box = wait.until(EC.presence_of_element_located(
                            (By.NAME, "p")))       
      except TimeoutException:
        print("Yahoo not loaded")
        continue

#assert 'Yahoo' in browser.title
      elem = browser.find_element_by_name('p')  # Find the search box
      elem.clear()
      browser.wait = WebDriverWait(browser, 4)
      elem.send_keys(company_name+'+bloomberg+private' + Keys.RETURN)   
      time.sleep(sleep_time)
      i=1
      j_all_urls=1
      url_df = url_df[0:0]
      url_df_mod = url_df_mod[0:0]
      for url in browser.find_elements_by_class_name("wr-bw"):    
        if "privcapId" in url.text: 
          url_df.at[i,'URL']=  url.text
          url_df.at[i,'UnqId']=  comp_names_indx
          url_df.at[i,'Company']=  company_name        
          i=i+1
    
        url_df_mod.at[j_all_urls,'URL']=  url.text
        url_df_mod.at[j_all_urls,'UnqId']=  comp_names_indx
        url_df_mod.at[j_all_urls,'Company']=  company_name 
        j_all_urls = j_all_urls+1
            
      if len(url_df) >0:       
       final_url_df.at[p_indx_url_exists,'URL']=  "https://"+url_df.loc[1,'URL']
       final_url_df.at[p_indx_url_exists,'Company']=  url_df.loc[1,'Company']
       final_url_df.at[p_indx_url_exists,'UnqId']=  url_df.loc[1,'UnqId']
       final_url_df.at[p_indx_url_exists,'Type']= comp_type       
       p_indx_url_exists= p_indx_url_exists+1
         
      url_df_final = url_df_final.append(url_df_mod)
    browser.quit()
    browser1=urllib2.build_opener()
    browser1.addheaders=[('User-agent', 'Mozilla/5.0')]
    comp_indx = 1
    website_url=''
    for comp_indx in xrange(1,len(final_url_df)+1):        
        sleep_time = random.randint(5,10)
        time.sleep(sleep_time)
        try:
            url_bloomberg = final_url_df.loc[comp_indx,'URL']
            response=browser1.open(url_bloomberg,timeout=(5.0))
            #Initializing Beautifulsoup object
            html = response.read()
            bsObj = BeautifulSoup(html)
            if final_url_df.loc[comp_indx,'Type'] == 'Target':
             website_list = bsObj.find("a", { "itemprop" : "url" })
             website_url = (website_list['href'] if str(website_list) <> 'None' else '')       
        except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            print 'ERROR FOR URL:',url_bloomberg
            print error_type, 'Line:', error_info.tb_lineno  
            continue

 
        #company_attrib_df_mod = getCompanyAttributes(final_url_df.loc[comp_indx,'Company'],final_url_df.loc[comp_indx,'UnqId'],bsObj,final_url_df.loc[comp_indx,'Type'],'')
        #company_attrib_df= company_attrib_df.append(company_attrib_df_mod)
        ls_indx_file = data_pd_final.loc[data_pd_final['Target Company']==final_url_df.loc[comp_indx,'Company']].index.values
        if len(ls_indx_file  )>0:
            indx_file = ls_indx_file.astype(int)[0]
            data_pd_final.at[indx_file,'URL']=website_url
        notselobj_ppl = bsObj.find("div",attrs={"class" : "fLeft tabPeople"}).find("a",attrs={"class" : "notSelected"})
        selobj_ppl = bsObj.find("div",attrs={"class" : "fLeft tabPeople"}).find("a",attrs={"class" : "selected"})
        try:
            people_tab_select_url = (notselobj_ppl['href'] if str(notselobj_ppl) <> 'None' else selobj_ppl['href']).replace("../","https://www.bloomberg.com/research/stocks/")
        except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            print 'ERROR People tab not found:'
            print error_type, 'Line:', error_info.tb_lineno
            continue    
        try:
            response1=browser1.open(people_tab_select_url,timeout=(5.0))
            sleep_time_ppl = random.randint(1,5)
            time.sleep(sleep_time_ppl) 
            html1 = response1.read()
            bsObj1 = BeautifulSoup(html1)
        except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            print 'ERROR FOR URL:',people_tab_select_url
            print error_type, 'Line:', error_info.tb_lineno
            continue
        #Initializing Beautifulsoup object
 
        if comp_indx == 1:
            people_df = getpeopleinfo(bsObj1,final_url_df.loc[comp_indx,'Company'],final_url_df.loc[comp_indx,'UnqId'])
            relation_df = getrelations(people_df,browser1)
        else:
            people_df_mod = getpeopleinfo(bsObj1,final_url_df.loc[comp_indx,'Company'],final_url_df.loc[comp_indx,'UnqId'])
            relation_df_mod = getrelations(people_df_mod,browser1)    
            people_df = people_df.append(people_df_mod)
            relation_df= relation_df.append(relation_df_mod)
    return data_pd_final,people_df,relation_df
  

#df_main_patents = pd_company_patent_info_final
#df_main_excel = df_excel     
def tech_features(df_main_patents,df_main_excel):
   repls = {',' : '', 'amp;':'','-': '','&': '','the':'','and':'','+':'','"':'','\'':''}

   df_main_excel_mod = df_main_excel 
   for i in xrange(1,len(df_main_excel)+1):     
    company_name = df_main_excel_mod.loc[i,'Target Company'].strip()
    company_name = company_name.lower()
    index= company_name.find('(')
    if index >0:
       df_main_excel_mod.at[i,'Cleaned_Company_Main'] = company_name[0:index-1]
    else:
       df_main_excel_mod.at[i,'Cleaned_Company_Main'] = company_name

   df_main_patents = df_main_patents.filter(['Company','Patent_Number','Num_Back_Citations','Num_Fwd_Citations','Priority_Date','Publication_Date','Assignee','Status'])
   df_main_patents=df_main_patents.loc[df_main_patents['Assignee'].notnull()]
   df_main_patents=df_main_patents.drop_duplicates()

   df_main_patents.reset_index()
   df_main_patents.index = np.arange(1, len(df_main_patents) + 1)
   df_main_patents['Cleaned_Company'] = map(lambda x: reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), x.lower().strip()),df_main_patents['Assignee'])
   #df_main_excel_mod['Cleaned_Company_Main'] = map(lambda x: reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), x.lower().strip()),df_main_excel_mod['Target Company'])

   df_main_excel_mod.reset_index()
   df_main_excel_mod.index = np.arange(1, len(df_main_excel_mod) + 1)
   repls_main = {',' : '', 'amp;':'','-': '','&': '','the':'','and':'','+':'','"':'','\'':''}
   #i=2
   for i in xrange(1,len(df_main_excel)+1):
    company = df_main_excel_mod.loc[i,'Target Company'].strip()
    company_name_cleaned = df_main_excel_mod.loc[i,'Cleaned_Company_Main']
    deal_year = df_main_excel_mod.loc[i,'Deal_Year']
    len_split_patent = len(company_name_cleaned.split(' '))
    len_split_main = len(company_name_cleaned.split(' '))
    if len(company.split(' '))>1:
     company_clnd_new =  reduce(lambda a, kv: a.replace(*kv), repls_main.iteritems(),company_name_cleaned)   
     patent_lst = df_main_patents.loc[df_main_patents['Cleaned_Company'].str.contains(company_clnd_new),'Patent_Number'].values
     indx_patents_ls  = df_main_patents.loc[df_main_patents['Cleaned_Company'].str.contains(company_clnd_new)].index.values
     Number_Patents = len(df_main_patents.loc[df_main_patents['Cleaned_Company'].str.contains(company_clnd_new)].index.values)  
    else:
     patent_lst = df_main_patents.loc[df_main_patents['Cleaned_Company']==company_name_cleaned,'Patent_Number'].values   
     indx_patents_ls  = df_main_patents.loc[df_main_patents['Cleaned_Company'] == company_name_cleaned].index.values
     Number_Patents = len(df_main_patents.loc[df_main_patents['Cleaned_Company']== company_name_cleaned].index.values)  
     
    nci_val_ls=[]    
    recent_indx=0
    nci_val=0
    #indx_fwd_cit =0
    for indx in indx_patents_ls:
     patent_pub_year = (pd.to_datetime(df_main_patents.loc[indx,'Publication_Date'],format='%Y-%m-%d').year if str(df_main_patents.loc[indx,'Publication_Date'])<> 'nan' else datetime.now().year)
     if len_split_main > 1 and company_name_cleaned.split(' ')[len_split_patent-1].lower().replace(' ','') == company_name_cleaned.split(' ')[len_split_main-1].lower().replace(' ',''):         
       if patent_pub_year >= deal_year-10:
          recent_indx=recent_indx+1
       num_fwd_cit =( df_main_patents.loc[indx,'Num_Fwd_Citations'] if str(df_main_patents.loc[indx,'Num_Fwd_Citations']) <> 'nan' else 0)
       nci_val_cal = 0.415*(math.pow(((datetime.datetime.now().year)-patent_pub_year+1),0.15)*num_fwd_cit)
       #print nci_val_cal 
       nci_val_ls.append(nci_val_cal)
       #indx_fwd_cit = indx_fwd_cit+1
     else:
       if patent_pub_year >= deal_year-10:
          recent_indx=recent_indx+1  
       num_fwd_cit = df_main_patents.loc[indx,'Num_Fwd_Citations']
       nci_val = 0.415*(math.pow(((datetime.now().year)-patent_pub_year+1),0.15)*num_fwd_cit)
       #print nci_val 
       nci_val_ls.append(nci_val_cal)
       #indx_fwd_cit = indx_fwd_cit+1
           
         
    nci_value = (sum(nci_val_ls) if len(nci_val_ls)>0 else 0)
    impact_patent = (nci_value/recent_indx if recent_indx <> 0 else 0)
    df_main_excel.at[i,'Number_Of_Patents'] = Number_Patents
    df_main_excel.at[i,'Number_Recent_Patents'] = recent_indx
    df_main_excel.at[i,'Impact_Of_Patents'] = impact_patent     
   return df_main_excel   
   
def patents_scrape(df_excel,path):
  profile = webdriver.FirefoxProfile()
  profile.set_preference("browser.download.folderList", 2)
  profile.set_preference("browser.download.manager.showWhenStarting", False)
  profile.set_preference("browser.download.dir", path)
  profile.set_preference("browser.download.downloadDir",path) 
  profile.set_preference("browser.download.defaultFolder",path)
  #profile.set_preference("browser.download.defaultDir","A:\\patents")
  profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

  browser_patent_sl = webdriver.Firefox(firefox_profile=profile,executable_path='A:\geckodriver-v0.19.0-win64\geckodriver.exe')
 
  time.sleep(15)
  mouse = webdriver.ActionChains(browser_patent_sl)

  data_final_company_pd = pd.DataFrame( columns = ['Company','Type','Patents'])
  #data_pd_comp = pd.DataFrame( columns = ['Company','Type'])
  pd_company_patent_info = pd.DataFrame( columns = ['Company','UnqId','Patent_Number','Type','Num_Back_Citations','Num_Fwd_Citations','Priority_Date','Publication_Date','Assignee','Title','Researchers','Status','URL'])
  fwd_citation_df = pd.DataFrame( columns = ['Company','UnqId','Patent_Number','Fwd_Citation','Examiner_Cited','Priority_Date','Publication_Date','Assignee','Title'])
  companies_patent_df = pd.DataFrame(columns = ['id','title','assignee','inventor/author','priority date','filing/creation date','publication date','grant date','result link'])
  pd_company_patent_info_final = pd.DataFrame( columns = ['Company','UnqId','Patent_Number','Type','Num_Back_Citations','Num_Fwd_Citations','Priority_Date','Publication_Date','Assignee','Title','Researchers','Status','URL'])
  fwd_citation_df_final = pd.DataFrame( columns = ['Company','UnqId','Patent_Number','Fwd_Citation','Examiner_Cited','Priority_Date','Publication_Date','Assignee','Title'])

  browser_patent=urllib2.build_opener()
  browser_patent.addheaders=[('User-agent', 'Mozilla/5.0')]
  df_excel.reset_index()
  df_excel.index = np.arange(1, len(df_excel) + 1)  
  
  data_final_company_pd = df_excel
  repls = {',' : '', 'amp;':'','.': '','-': '','&': '','co.':''}

  i_all_ptnt_inx = 1
  i_comp_indx_patent=1

#data_final_company_pd.loc[data_final_company_pd['Company']=='PLx Pharma Inc.'].index.values.astype(int)[0]

  for i_comp_indx_patent in xrange(1,len(data_final_company_pd)+1):
   sleep_time = random.randint(6,10)
   random_number = random.randint(5,12)  
   #company_name = 'PLx Pharma Inc.'
   company_name  =  data_final_company_pd.loc[i_comp_indx_patent,'Target Company'] 
   company_name_clean = reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), company_name.lower().strip()).strip()
   company_name_clean = company_name_clean.replace(' ','+')
   url_assign = 'https://patents.google.com/?assignee='+company_name_clean
   time.sleep(5)
   browser_patent_sl.get(url_assign)
   time.sleep(sleep_time)
   #path = "A:/patents/"

  
   filelist_del = glob.glob(os.path.join(path+'\\', "*.csv"))
   for f_del in filelist_del:
    os.remove(f_del)

   try:
     wait = WebDriverWait(browser_patent_sl, 15)
     box = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "#count a")))
     browser_patent_sl.find_element_by_css_selector("#count a").click()  
       
   except Exception:
    #df_excel.at[i_comp_indx_patent,'Patents'] = 'N' 
    print("Patent not found")    
    continue 
   time.sleep(10)
   all_files = glob.glob(os.path.join(path+'\\', "*.csv")) #make list of paths
   i_patent_files = 1
 #read files
   for file in all_files:
    if i_patent_files   ==1:
     companies_patent_df = pd.read_csv(file,skiprows=1)
   i_patent_files=i_patent_files+1 
   companies_patent_df.reset_index()
   companies_patent_df.index = np.arange(1, len(companies_patent_df) + 1)

   i_patent_info =1

   for i_patent_info in xrange(1,len(companies_patent_df)+1):
    fetch_sleep = random.randint(5,10)   
    fwd_cit_indx = len(fwd_citation_df)+1
    i_all_ptnt_inx = len(pd_company_patent_info)+1
    url_patents =''
    pd_company_patent_info.at[i_all_ptnt_inx,'Patent_Number'] = companies_patent_df.loc[i_patent_info,'id']
    pd_company_patent_info.at[i_all_ptnt_inx,'Company'] = company_name
    pd_company_patent_info.at[i_all_ptnt_inx,'Assignee'] = companies_patent_df.loc[i_patent_info,'assignee'] 
    pd_company_patent_info.at[i_all_ptnt_inx,'Publication_Date'] = companies_patent_df.loc[i_patent_info,'publication date']
    pd_company_patent_info.at[i_all_ptnt_inx,'Title'] = companies_patent_df.loc[i_patent_info,'title'].encode('utf8')
    pd_company_patent_info.at[i_all_ptnt_inx,'Researchers'] = companies_patent_df.loc[i_patent_info,'inventor/author']
    pd_company_patent_info.at[i_all_ptnt_inx,'Priority_Date'] = companies_patent_df.loc[i_patent_info,'priority date']  
    pd_company_patent_info.at[i_all_ptnt_inx,'URL'] = companies_patent_df.loc[i_patent_info,'result link']          

    #print companies_patent_df.loc[i_patent_info,'assignee'] 
    #fwd_citation_df[0:0]
    #bckwd_citation_df[0:0]
    try:
      url_patents = companies_patent_df.loc[i_patent_info,'result link']
      #url_patents='https://patents.google.com/patent/US20080050738A1/en'
      response_patent=browser_patent.open(url_patents,timeout=(5.0))
      time.sleep(5)
      #Initializing Beautifulsoup object
      html_patent = response_patent.read()
      bsObj_patent = BeautifulSoup(html_patent)
    except Exception as e:
     error_type, error_obj, error_info = sys.exc_info()
     print 'ERROR FOR URL:',url_patents
     print error_type, 'Line:', error_info.tb_lineno    
     continue
    obj_legal_status = bsObj_patent.find("dd", { "itemprop" : "applicationNumber" })
    obj_inventor = bsObj_patent.findAll("dd", { "itemprop" : "inventor" })
    
    pd_company_patent_info.at[i_all_ptnt_inx,'Status'] = (bsObj_patent.find("span", { "itemprop" : "status" }).text
                                                              if str(bsObj_patent.find("span", { "itemprop" : "status" })) <> 'None' else '')                                                      
    
    #['Company','UnqId','Patent_Number','Type','Num_Back_Citations','Num_Fwd_Citations','Publication_Date','Assignee','Title','Researchers'])
   
    
    table_obj_fwd = bsObj_patent.findAll("tr", attrs={ "itemprop" : "forwardReferences" })
    i_ptnt_fwd_indx = 0
    
    if str(table_obj_fwd) <> 'None':
     for row in table_obj_fwd:         
         cells = row.findAll("td")
         if len(cells) >1 :
           fwd_cit_indx=fwd_cit_indx+1
           i_ptnt_fwd_indx=i_ptnt_fwd_indx+1 
    pd_company_patent_info.at[i_all_ptnt_inx,'Num_Fwd_Citations'] = i_ptnt_fwd_indx
    i_all_ptnt_inx=i_all_ptnt_inx+1
    
   pd_company_patent_info_final = pd_company_patent_info_final.append(pd_company_patent_info)
   pd_company_patent_info_final.reset_index()
   pd_company_patent_info_final.index = np.arange(1, len(pd_company_patent_info_final) + 1)
   browser_patent_sl.quit()
   if len(pd_company_patent_info_final)>0:
    df_excel = tech_features(pd_company_patent_info_final,df_excel)
   
  return df_excel

def people_score(input_file,people,relationship):
		people.reset_index()
                people.index = np.arange(1, len(people) + 1) 
                
                relationship.reset_index()
                relationship.index = np.arange(1, len(relationship) + 1)   

		#Creating the Nodes for the network.
		p1 = list(pd.unique(relationship.Person_Name))
		p2 = list(pd.unique(relationship.Relation_Name))
		person_nodes = list(pd.unique(p1+p2))

		#Graph Building

		G = nx.Graph()
		G.add_nodes_from(person_nodes)

		dummy1 = []

		print "Building Graph started..."

		for i in person_nodes:
			if (i in list(people.Person_Name)) & (i not in dummy1):
				dummy1.append(i)
				comp_list = list(pd.unique((people[people.Person_Name == i]['Company_Name'])))
				title_list = list(pd.unique((people[people.Person_Name == i]['Title'])))
				comp_list2 = list(pd.unique((people[people.Person_Name == i]['Primary_Company'])))
				G.nodes[i]['Primary Company'] = comp_list[0]
				G.nodes[i]['Title'] = title_list[0]
				G.nodes[i]['Secondry Comp'] = comp_list2[0]        
				G.nodes[i]['From'] = 'Primary Person'

		print "Building Graph 1st Loop done..."

		for i in person_nodes:
			if (i not in dummy1) & (i in list(relationship.Relation_Name)):
				dummy1.append(i)
				comp_list = list(pd.unique((relationship[relationship.Relation_Name == i]['Relation_Company'])))
				G.nodes[i]['Primary Company'] = comp_list[0]
				G.nodes[i]['From'] = 'Rshp Person'
				
		print "Building Graph 2nd Loop done..."

		for i in range(1,len(relationship)+1):
			G.add_edge(relationship.Person_Name[i],relationship.Relation_Name[i])
			
		print "Building Graph Completed..."

		print ">>> No. of Nodes/People.....:", G.number_of_nodes(),"\n>>> No. of Edges/Connections:",G.number_of_edges()

		nodes = G.node

		for t in input_file['Target Company']:
			counter = [0]
			for i in  G.nodes():
				if (nodes[i]['Primary Company'] == t):
					try:
						counter.append(nx.degree(G)[i])
					except:
						counter.append(int(0))
				a = max(counter)
			input_file.loc[input_file['Target Company']==t,'People'] = a 

		#input_file.to_excel("D:/ISB/Capstone Project/People_Network_Output.xlsx")
		print "People Network Analysis completed..."	
		return input_file
		
def draw_network_person(p):
		if p in list(G.node):
			list1 = list(G[p])
			list2 = []
			for i in list1:
				temp_list = list(G[i])
				list2 += temp_list
			list3 = list1+list2+[p]

			H = G.subgraph(list3)
			nx.draw(H,with_labels=True)
			plt.show()
		else:
			print "Person Name Not found in Network"
		
	# person_name = raw_input("Enter a person name for his netwrok graph:")
	# draw_network_person(person_name) 

def draw_network_company(c):
		list1 = []
		for i in person_nodes:
			if nodes[i]['Primary Company'] == c:
				list1 += list(G[i])
			else:
				print "Company name not found."
		H = G.subgraph(list1)
		nx.draw(H,with_labels=True)
		plt.show()
		
df_excel = pd.read_excel('A:/Capstone_Code/Modelling/Input_File.xlsx')
path = "A:\\patents"
df_excel['Number_Of_Patents']=0
df_excel['Number_Recent_Patents']=0
df_excel['Impact_Of_Patents']=0
df_excel['People']=0
df_excel['Number_Fav_Twitter']=0
df_excel['Sentiment_Score']=0
CompanyURLs= pd.DataFrame(columns=['company','website'])
df_excel,people_df,relation_df = getCompanyAttributes(df_excel)
df_excel = patents_scrape(df_excel,path)
df_excel= people_score(df_excel,people_df,relation_df)
CompanyURLs = get_twitter_url(df_excel,CompanyURLs)
path = path+"\\"
df_excel = get_tweets_text(df_excel,CompanyURLs,path)
writer = pd.ExcelWriter('Modelling_Final_File_Imputation.xlsx', engine='xlsxwriter')
df_excel.to_excel(writer,'Sheet1')
writer.save()
