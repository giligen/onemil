# Stage E — the `news_only` headlines (F8 N=5, 30 random TRAIN trades of the bucket)

generated 2026-09-17 00:43:50 | window prev-day 15:00 ET -> 09:30 ET, the same endpoint and window `D/d0_news.py` used | classifier fixed before the pull

symbol-days sampled: 30, with at least one premarket article: 30, articles: 54

## class mix (per article)

cls
other    31
event    14
recap     6
both      3

## per symbol-day: the dominant class and that trade's net R

       day symbol  n   cls     net
2025-01-23   LTBR  1 other 12.8599
2025-02-03     JG  1 recap  1.4013
2025-02-05   CRNC  1 recap  1.0362
2025-02-11   PTON  1 other -1.0922
2025-02-14   ALKS  1 other -1.2640
2025-03-04     AA  1 event -0.5018
2025-03-11   ADPT  1 other  2.1498
2025-03-11    RKT  1 other -0.0312
2025-03-12    TLN  1 event -1.1077
2025-03-17   MNDY  2 other -1.2750
2025-04-03   ADPT  1 recap -0.8322
2025-04-03   HSAI  1 event -0.3652
2025-04-17   RIVN  2 event -1.3037
2025-04-23     MC  1 event -1.1993
2025-05-01   MRAM  3 other  2.3943
2025-06-27   JFBR  1 event -1.0503
2025-07-02    GBX  8 other  0.5964
2025-07-30   LUNR  1 event -1.2215
2025-07-31     AX  4 other -1.1924
2025-08-01   NSIT  1 other -0.5574
2025-08-04    NET  1 other  0.0976
2025-08-06   AZTA  1 event -0.5689
2025-08-07    SUZ  1 other  0.5488
2025-08-13   PUBM  3 other -1.1413
2025-09-08   PAAS  1 event -0.0340
2025-09-18   PESI  1  both -1.0630
2025-10-30   AXGN  2 other -0.4403
2025-11-07   PTON  7 other  0.6828
2025-12-10   GIII  2 other  0.2240
2025-12-11   ARRY  1 other  2.7465

## mean net R of the sampled trades by dominant class

       size    mean
cls                
both      1 -1.0630
event     9 -0.8169
other    17  0.9004
recap     3  0.5351

## every headline

       day symbol                   ts   cls                                                                                                                                                                                                                           headline
2025-04-23     MC 2025-04-23T08:32:28Z event                                                                                                                                                                                              Earnings Scheduled For April 23, 2025
2025-02-03     JG 2025-02-03T12:05:32Z recap                                                                                                                                                             12 Information Technology Stocks Moving In Monday's Pre-Market Session
2025-08-07    SUZ 2025-08-07T08:04:23Z other                                                                                                                                                           Suzano Q2 EPS $0.71 Up From $(0.56) YoY, Sales $2.35B Up From $2.21B YoY
2025-05-01   MRAM 2025-04-30T20:05:43Z other                                                                                                                                                  Everspin Technologies Q2 GAAP EPS expected to be more than $(0.05) vs $(0.01) Est
2025-05-01   MRAM 2025-04-30T20:04:35Z other                                                                                                          Everspin Technologies Q2  Adj EPS expected to be below $0.05 vs $0.04 Est; Sees Q2 Sales $12.500M-$13.500M vs $12.90M Est
2025-05-01   MRAM 2025-04-30T20:03:00Z other                                                                                                                                   Everspin Technologies Q1 Adj. EPS $0.02 Up From $(0.01) YoY, Sales $13.14M Beat $12.50M Estimate
2025-02-05   CRNC 2025-02-05T13:20:11Z recap                                                                                                                                                  Jim Cramer: This Health Care Stock Is A 'Winner,' Buy Bitcoin Instead Of Coinbase
2025-07-02    GBX 2025-07-02T12:06:49Z  both                                                                                                                                                                     12 Industrials Stocks Moving In Wednesday's Pre-Market Session
2025-07-02    GBX 2025-07-02T11:45:45Z recap                                                                                                                                                                                                    Market-Moving News for July 2nd
2025-07-02    GBX 2025-07-02T10:01:18Z other                                                                                                                                      US Stocks Likely To Open Higher: S&P 500 Sees 'Average Gain Of 6.1%' In 2nd Half, Expert Says
2025-07-02    GBX 2025-07-02T04:38:17Z recap                                                                                                                                                        Constellation Brands, UniFirst And 3 Stocks To Watch Heading Into Wednesday
2025-07-02    GBX 2025-07-02T04:16:46Z other                                                                                                           Dow Jumps 400 Points As Senate Approves Tax Bill: Investor Sentiment Edges Lower, But Fear Index Remains In 'Greed' Zone
2025-07-02    GBX 2025-07-01T21:05:42Z  both                                                                                                                                                                     12 Industrials Stocks Moving In Tuesday's After-Market Session
2025-07-02    GBX 2025-07-01T20:19:18Z event                                                                                                                                                  Greenbrier Companies Affirms FY2025 Sales Guidance of $3.15B-$3.35B vs $3.22B Est
2025-07-02    GBX 2025-07-01T20:18:04Z other                                                                                                                                      Greenbrier Companies Q3 EPS $1.86 Beats $0.98 Estimate, Sales $842.70M Beat $785.72M Estimate
2025-04-17   RIVN 2025-04-17T10:57:53Z event                                                                                                                             Rivian Delivers Vehicles To HelloFresh In First Non-Amazon Deal After Ending Its Exclusive Partnership
2025-04-17   RIVN 2025-04-16T20:21:10Z other                                                                                                                                                                                          What's Going On With Rivian (RIVN) Stock?
2025-07-30   LUNR 2025-07-30T12:31:39Z event                                                                           Intuitive Machines Announced It Has Secured A $9.8M Phase Two Government Contract To Advance Its Orbital Transfer Vehicle Through Critical Design Review
2025-03-17   MNDY 2025-03-17T13:00:59Z other                                                                                                                                                                           Beyond The Numbers: 25 Analysts Discuss Monday.Com Stock
2025-03-17   MNDY 2025-03-17T11:25:44Z event                                                                                                                                                             DA Davidson Upgrades Monday.Com to Buy, Maintains Price Target to $350
2025-04-03   HSAI 2025-04-03T12:36:07Z event     Hesai Technology's Lidar Solution Selected By WeRide To Power Autonomous Vehicles On Uber's Platform In Dubai, Supporting The City's 2030 Smart Mobility Initiative And Expanding Autonomous Transportation In The Middle East
2025-08-13   PUBM 2025-08-13T12:30:23Z other                                                                                                                                                    Evercore ISI Group Maintains Outperform on PubMatic, Lowers Price Target to $12
2025-08-13   PUBM 2025-08-12T20:00:07Z event                                                                                                                                                                  These Analysts Slash Their Forecasts On PubMatic After Q2 Results
2025-08-13   PUBM 2025-08-12T19:13:44Z other                                                                                                                                                       Micron To Rally More Than 49%? Here Are 10 Top Analyst Forecasts For Tuesday
2025-07-31     AX 2025-07-31T13:01:03Z other                                                                                                                                                                                          Where Axos Financial Stands With Analysts
2025-07-31     AX 2025-07-31T11:36:59Z other                                                                                                                                     Keefe, Bruyette & Woods Maintains Market Perform on Axos Financial, Raises Price Target to $94
2025-07-31     AX 2025-07-31T10:52:34Z other                                                                                                                                                               Needham Maintains Buy on Axos Financial, Raises Price Target to $102
2025-07-31     AX 2025-07-30T20:20:22Z other                                                                                                                                       Axos Financial Q4 Adj. EPS $1.94 Beats $1.81 Estimate, Sales $321.45M Beat $312.29M Estimate
2025-06-27   JFBR 2025-06-26T21:05:55Z event                                                                                                                                     Jeffs Brands Files Prospectus For Resale Of Up To 52.5M Ordinary Shares By Selling Stockholder
2025-08-04    NET 2025-08-04T11:32:23Z other                                                                                                                                                      5 Stocks In The Spotlight From Wall Street's Most Accurate Analysts Last Week
2025-10-30   AXGN 2025-10-30T13:08:49Z other                                                                                                                                                              Canaccord Genuity Maintains Buy on Axogen, Raises Price Target to $27
2025-10-30   AXGN 2025-10-30T12:53:46Z other                                                                                                                                                         Citizens Maintains Market Outperform on Axogen, Raises Price Target to $34
2025-08-06   AZTA 2025-08-06T11:29:49Z event                                                                                                                                                            Raymond James Upgrades Azenta to Outperform, Announces $35 Price Target
2025-09-18   PESI 2025-09-18T12:09:18Z  both                                                                                                                                                                      12 Industrials Stocks Moving In Thursday's Pre-Market Session
2025-11-07   PTON 2025-11-07T13:48:16Z other                                                                                                                                   Telsey Advisory Group Maintains Market Perform on Peloton Interactive, Maintains $9 Price Target
2025-11-07   PTON 2025-11-07T07:21:50Z other                                                                                                                                                                  Peloton Recalls Nearly 878,000 Exercise Bikes Over Breaking Seats
2025-11-07   PTON 2025-11-06T22:53:44Z event                                                                                                                            Peloton Interactive Raises FY2026 Sales Guidance from $2.400B-$2.500B to $2.491B-$2.500B vs $2.454B Est
2025-11-07   PTON 2025-11-06T22:48:46Z event                                                                                                                                                                                Peloton Stock Rallies After Q1 Earnings: Here's Why
2025-11-07   PTON 2025-11-06T21:06:37Z recap                                                                                                                                                         12 Consumer Discretionary Stocks Moving In Thursday's After-Market Session
2025-11-07   PTON 2025-11-06T21:04:16Z other                                                                                                                                                             Peloton Interactive Sees Q2 Sales $665.000M-$685.000M vs $664.634M Est
2025-11-07   PTON 2025-11-06T21:03:33Z other                                                                                                                                     Peloton Interactive Q1 EPS $0.03 Beats $0.01 Estimate, Sales $550.800M Beat $539.816M Estimate
2025-04-03   ADPT 2025-04-02T21:05:37Z recap                                                                                                                                                                   12 Health Care Stocks Moving In Wednesday's After-Market Session
2025-12-11   ARRY 2025-12-10T20:57:20Z other                                                                                                                                                                       Wall Street Rallies, Shrugs Off Powell's Wait-And-See Stance
2025-12-10   GIII 2025-12-10T13:43:23Z other                                                                                                                                                    Keybanc Maintains Overweight on G-III Apparel Group, Raises Price Target to $35
2025-12-10   GIII 2025-12-10T12:26:26Z other                                                                                                                                  Telsey Advisory Group Maintains Market Perform on G-III Apparel Group, Raises Price Target to $34
2025-03-12    TLN 2025-03-12T09:58:42Z event                                                                                                                           Morgan Stanley Initiates Coverage On Talen Energy with Overweight Rating, Announces Price Target of $243
2025-02-11   PTON 2025-02-10T21:14:12Z other                                                                                                                                                      Macquarie Maintains Neutral on Peloton Interactive, Maintains $9 Price Target
2025-03-11    RKT 2025-03-10T22:33:49Z other                                                                                                                                  Rocket, Redfin Deal Aims To Create End-To-End Real Estate Solution: Where Does That Leave Zillow?
2025-09-08   PAAS 2025-09-08T10:34:57Z event Pan American Silver Reports Drill Results For La Colorada Mine In Zacatecas, Mexico, Incl. Multiple High-Grade Veins Indicating Potential For Expansion Of Silver Mineral Resources, Extension Of Mine Life And Improved Economics
2025-02-14   ALKS 2025-02-14T11:42:52Z other                                                                                                                                                                Goldman Sachs Maintains Buy on Alkermes, Raises Price Target to $32
2025-08-01   NSIT 2025-07-31T19:00:50Z other                                                                                                                                                                              What Does the Market Think About Insight Enterprises?
2025-03-04     AA 2025-03-03T21:45:17Z event                                                                                                                                                                            Alcoa Subsidiary Prices Offering Of $1B Of Senior Notes
2025-03-11   ADPT 2025-03-11T12:06:48Z other                                                                                            Adaptive Biotechnologies Announces Enhanced clonoSEQ Assay For MRD Detection In DLBCL Using ctDNA; Achieves 7-Fold Sensitivity Increase
2025-01-23   LTBR 2025-01-23T11:20:21Z other                                                                                                                                Lightbridge And Oklo Enter Non-binding Memorandum Of Understanding Regarding Collaboration - Filing
