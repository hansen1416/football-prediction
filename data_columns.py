columns_basic = ['PlayerUrl', 'PlayerName', 'Date', 'Day', 'Comp', 'Round', 'Venue',
                 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min']

columns_summary = ['Summery-Performance-Gls', 'Summery-Performance-Ast', 'Summery-Performance-PK', 'Summery-Performance-PKatt', 'Summery-Performance-Sh', 'Summery-Performance-SoT', 'Summery-Performance-CrdY', 'Summery-Performance-CrdR', 'Summery-Performance-Touches', 'Summery-Performance-Press', 'Summery-Performance-Tkl', 'Summery-Performance-Int',
                   'Summery-Performance-Blocks', 'Summery-Expected-xG', 'Summery-Expected-npxG', 'Summery-Expected-xA', 'Summery-SCA-SCA', 'Summery-SCA-GCA', 'Summery-Passes-Cmp', 'Summery-Passes-Att', 'Summery-Passes-Cmp%', 'Summery-Passes-Prog', 'Summery-Carries-Carries', 'Summery-Carries-Prog', 'Summery-Dribbles-Succ', 'Summery-Dribbles-Att']

columns_passing = ['Passing-Total-Cmp', 'Passing-Total-Att', 'Passing-Total-Cmp%', 'Passing-Total-TotDist', 'Passing-Total-PrgDist', 'Passing-Short-Cmp', 'Passing-Short-Att', 'Passing-Short-Cmp%', 'Passing-Medium-Cmp',
                   'Passing-Medium-Att', 'Passing-Medium-Cmp%', 'Passing-Long-Cmp', 'Passing-Long-Att', 'Passing-Long-Cmp%', 'Passing-Ast', 'Passing-xA', 'Passing-KP', 'Passing-1/3', 'Passing-PPA', 'Passing-CrsPA', 'Passing-Prog']


columns_passing_types = ['PassingTypes-PassAttempted', 'PassingTypes-PassTypes-Live', 'PassingTypes-PassTypes-Dead', 'PassingTypes-PassTypes-FK', 'PassingTypes-PassTypes-TB', 'PassingTypes-PassTypes-Press', 'PassingTypes-PassTypes-Sw', 'PassingTypes-PassTypes-Crs', 'PassingTypes-PassTypes-CK', 'PassingTypes-CornerKicks-In', 'PassingTypes-CornerKicks-Out', 'PassingTypes-CornerKicks-Str',
                         'PassingTypes-Height-Ground', 'PassingTypes-Height-Low', 'PassingTypes-Height-High', 'PassingTypes-BodyParts-Left', 'PassingTypes-BodyParts-Right', 'PassingTypes-BodyParts-Head', 'PassingTypes-BodyParts-TI', 'PassingTypes-BodyParts-Other', 'PassingTypes-Outcomes-Cmp', 'PassingTypes-Outcomes-Off', 'PassingTypes-Outcomes-Out', 'PassingTypes-Outcomes-Int', 'PassingTypes-Outcomes-Blocks']

columns_gca = ['GCA-SCATypes-SCA', 'GCA-SCATypes-PassLive', 'GCA-SCATypes-PassDead', 'GCA-SCATypes-Drib', 'GCA-SCATypes-Sh', 'GCA-SCATypes-Fld', 'GCA-SCATypes-Def',
               'GCA-GCATypes-GCA', 'GCA-GCATypes-PassLive', 'GCA-GCATypes-PassDead', 'GCA-GCATypes-Drib', 'GCA-GCATypes-Sh', 'GCA-GCATypes-Fld', 'GCA-GCATypes-Def']


columns_defense = ['Defence-Tackles-Tkl', 'Defence-Tackles-TklW', 'Defence-Tackles-Def-3rd', 'Defence-Tackles-Mid-3rd', 'Defence-Tackles-Att-3rd', 'Defence-VsDribbles-Tkl', 'Defence-VsDribbles-Att', 'Defence-VsDribbles-Tkl%', 'Defence-VsDribbles-Past', 'Defence-Pressures-Press',
                   'Defence-Pressures-Succ', 'Defence-Pressures-%', 'Defence-Pressures-Def-3rd', 'Defence-Pressures-Mid-3rd', 'Defence-Pressures-Att-3rd', 'Defence-Blocks-Blocks', 'Defence-Blocks-Sh', 'Defence-Blocks-ShSv', 'Defence-Blocks-Pass', 'Defence-Int', 'Defence-Tkl+Int', 'Defence-Clr', 'Defence-Err']


columns_possession = ['Possession-Touches-Touches', 'Possession-Touches-Def-Pen', 'Possession-Touches-Def-3rd', 'Possession-Touches-Mid-3rd', 'Possession-Touches-Att-3rd', 'Possession-Touches-Att-Pen', 'Possession-Touches-Live', 'Possession-Dribbles-Succ', 'Possession-Dribbles-Att', 'Possession-Dribbles-Succ%', 'Possession-Dribbles-#Pl', 'Possession-Dribbles-Megs',
                      'Possession-Carries-Carries', 'Possession-Carries-TotDist', 'Possession-Carries-PrgDist', 'Possession-Carries-Prog', 'Possession-Carries-1/3', 'Possession-Carries-CPA', 'Possession-Carries-Mis', 'Possession-Carries-Dis', 'Possession-Receiving-Targ', 'Possession-Receiving-Rec', 'Possession-Receiving-Rec%', 'Possession-Receiving-Prog']

columns_misc = ['Misc-Performance-CrdY', 'Misc-Performance-CrdR', 'Misc-Performance-2CrdY', 'Misc-Performance-Fls', 'Misc-Performance-Fld', 'Misc-Performance-Off', 'Misc-Performance-Crs', 'Misc-Performance-Int',
                'Misc-Performance-TklW', 'Misc-Performance-PKwon', 'Misc-Performance-PKcon', 'Misc-Performance-OG', 'Misc-Performance-Recov', 'Misc-AerialDuels-Won', 'Misc-AerialDuels-Lost', 'Misc-AerialDuels-Won%']

columns_summary_short = ['Summery-Performance-Gls', 'Summery-Performance-Ast', 'Summery-Performance-PK', 'Summery-Performance-PKatt', 'Summery-Performance-Sh', 'Summery-Performance-SoT', 'Summery-Performance-CrdY',
                         'Summery-Performance-CrdR', 'Misc-Performance-Fls', 'Misc-Performance-Fld', 'Misc-Performance-Off', 'Misc-Performance-Crs', 'Misc-Performance-TklW', 'Misc-Performance-Int', 'Misc-Performance-OG', 'Misc-Performance-PKwon', 'Misc-Performance-PKcon']

if __name__ == "__main__":
    result = list(map(lambda x: x.replace(" ", "-"), columns_summary_short))

    print(result)
