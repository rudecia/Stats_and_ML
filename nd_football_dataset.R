library(tidyverse)


football <- read.csv("~/Downloads/R Homework and Datasets/nd_football.csv")

library(tidyverse)



#logical vector for ND wins
w <- sum(football$ND_score > football$Opp_score)


l <- sum(football$ND_score < football$Opp_score)

t <- sum(football$ND_score  ==football$Opp_score)

#341 losses, 978 wins, 42 ties

football <- football %>% 
  mutate(score_diff = football$ND_score - football$Opp_score) %>% 
  arrange(by = desc(score_diff))

football[1,]$Opponent

ggplot(data = football) + geom_point(mapping = aes(x = Year, y = score_diff), color = 'blue')

score_diff_tab <- football %>% 
  group_by(by = Year) %>% 
  summarise(m = mean(score_diff), med = median(score_diff))

#Plot the median score difference -- maybe add a horizontal line for 0?
ggplot(data = score_diff_tab) + geom_line(mapping = aes(x = by, y = med)) + 
  geom_point(mapping = aes(x = by, y = med), color = 'blue')


scoring <- function(num){
  if(num > 0) {
    return('W')
  }else if(num == 0){
    return('T')
  }else{
    return('L')}
  }

football <- football %>% 
  mutate(outcome = factor(unlist(map(score_diff, scoring)), levels = c('L', 'T', 'W')))


coach_summary <- football %>% 
  filter(ND.Coach != 'NO COACH') %>% 
  group_by(by = ND.Coach) %>% 
  summarise(win_pct = sum(outcome == 'W')/n(), num_games = n(), num_szns = max(Year) - min(Year) + 1)%>% 
  arrange(by = desc(win_pct)) %>% 
  ungroup()

#schools that have a winning record against ND football
#that have played at least ten games against our team
nd_killers <- football %>% 
  group_by(by = Opponent) %>%
  summarise(win_pct = sum(outcome == 'W')/n(), num_games = n()) %>% 
  filter(num_games >= 10, win_pct < 0.5) %>% 
  arrange(by = desc(win_pct))


#schools that have a losing record against ND
#after having played at least ten games against the team
nd_breakfast <- football %>% 
  group_by(by = Opponent) %>%
  summarise(win_pct = sum(outcome == 'W')/n(), num_games = n()) %>% 
  filter(num_games >= 10, win_pct > 0.5) %>% 
  arrange(by = desc(win_pct))

#the only school with a tied record and > 10 games is florida state
#after how last year went it doesn't look like this will remain the case for long
football %>% 
  group_by(by = Opponent) %>%
  summarise(win_pct = sum(outcome == 'W')/n(), num_games = n()) %>% 
  filter(num_games >= 10, win_pct == 0.5)



#Do winning coaches stay longer?
long_term_coaches <- coach_summary %>% 
  filter(num_szns > 2)

cor(tibble(long_term_coaches$win_pct,long_term_coaches$num_szns))

ggplot(data = long_term_coaches) +geom_point(mapping = aes(x = win_pct, y = num_szns))


