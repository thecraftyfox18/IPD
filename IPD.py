import axelrod as axl
import numpy as np
import matplotlib.pyplot as plt

reac_play = (0.8, 0.2)
prob = 0.2
type_s= ['Reactive', 'Non-Reactive']
num_match = 20
noise = [0.001,0.01,0.1]

def graph(counter,l):
    fig, ax = plt.subplots()
    im = ax.imshow(counter)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Numero di volte che il punteggio è ottimizzato", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(strategy)))
    ax.set_yticks([0])
    ax.set_xticklabels(strategy)
    ax.set_yticklabels([strategy[l]])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for m in range(len(strategy)):
        text = ax.text(m, 0, counter[0,m], ha="center", va="center", color="w")
    ax.set_title("Numero di ottimizzazioni per noise = %1.3f" % noise_)
    fig.tight_layout()
    plt.show()

print("\n")
print("Welcome to MLIPD, a Machine Learning program for the Iterated Prisoner's Dilemma :)\n")
print("You will create a series of games with a fictious prisoner and then the machine learning program will tell you what's the best strategy to play.\n")
print("First of all, do you wanna play reactive or non reactive strategy?")
print(0,type_s[0])
print(1,type_s[1])
choice=input("Your choice (number):\n")
choice=int(choice)
if choice==0:
    player_A = player_B= [axl.TitForTat(), axl.Grudger(), axl.BackStabber(), axl.GradualKiller(), axl.Alexei(),
                axl.ReactivePlayer(reac_play), axl.Desperate(), axl.Hopeless(), axl.Willing(), axl.NaiveProber(prob),
                axl.EasyGo()]
    strategy = ['TitForTat', 'Grudger', 'BackStabber', 'GradualKiller', 'Alexei', 'ReactivePlayer', 'Desperate',
                'Hopeless', 'Willing', 'NaiveProber', 'EasyGo']
elif choice==1:
    player_A = player_B = [axl.Alternator(),axl.Random(),axl.Defector(),axl.Cooperator(),axl.Cycler('CDDC')]
    strategy = ['Alternator','Random','Defector','Cooperator','Cycler']
else:
    choice=0
    player_A = player_B = [axl.TitForTat(), axl.Grudger(), axl.BackStabber(), axl.GradualKiller(), axl.Alexei(),
                           axl.ReactivePlayer(reac_play), axl.Desperate(), axl.Hopeless(), axl.Willing(),
                           axl.NaiveProber(prob),
                           axl.EasyGo()]
    strategy = ['TitForTat', 'Grudger', 'BackStabber', 'GradualKiller', 'Alexei', 'ReactivePlayer', 'Desperate',
                'Hopeless', 'Willing', 'NaiveProber', 'EasyGo']
    print('Invalid choice! Reactive strategy choose as default')
if choice==0:
    print("Please select the number of the strategy you want to play against:")
else:
    print("Please select the number of the strategy you want to play:")
for i in range(len(strategy)):
    print(i,strategy[i])
l=input("Your choice:\n")
l=int(l)
if l<0 or l>=len(strategy):
    l=0
    print("Error! First strategy set as default")
if choice==0:
    print("You are playing against:")
else:
    print("You are playing:")
print(strategy[l])
num_repetitions= input("How many games do you wanna play? (max value is 500)\n")
num_repetitions=int(num_repetitions)
if num_repetitions<=0:
    print("The number of games must be greater than 0!")
    num_repetitions=abs(num_repetitions)
elif num_repetitions>=500:
    print("The number of games must be smaller than 500! 100 is set as default")
    num_repetitions=100
print("Choose the noise that will affect your games")
for i in range(len(noise)):
    print(i,noise[i])
j=input("Your choice:\n")
j=int(j)
if j<0 or j>=len(noise):
    j=0
    print("Error! First noise set as default")
noise_=noise[j]
print(noise_)
print("Do you wanna see the number of optimization on a graph? (1 YES, 0 NO)")
ans=input("Your choice:\n")
ans=int(ans)
if ans!=0 and ans!=1:
    ans=0
    print("Invalid choice! The graph will not be shown.")
print("Creation of testing set.")
t = open("test.dat", "w")
counter = np.zeros((1,len(player_B)), dtype= int)
for k in range(len(player_B)):
    players = (player_A[l], player_B[k])
    match = axl.Match(players=players, turns=num_match, noise=noise_)
    game = []
    single_match = []
    resA = []
    resB = []
    for j in range(num_repetitions):
        match.play()
        if choice==0:
            if np.abs(match.final_score()[0] - match.final_score()[1]) <= 5:
                counter[0][k] += 1
        if choice==1:
            if match.final_score()[0] >match.final_score()[1]:
                counter[0][k] += 1
        for i in range(num_match):
            if match.result[i][0].value == 0:
                if match.result[i][1].value == 0:
                    single_match.append((1, 1))
                else:
                    single_match.append((1, -1))
            else:
                if match.result[i][1].value == 0:
                    single_match.append((-1, 1))
                else:
                    single_match.append((-1, -1))
        resA.append(match.final_score()[0])
        resB.append(match.final_score()[1])
        game.append(single_match)
        single_match = []
    for j in range(num_repetitions):
        if choice==0:
            if abs(resA[j] - resB[j]) <= 5:
                index = 1
            else:
                index = 0
        if choice==1:
            if resA[j]>resB[j]:
                index = 1
            else:
                index = 0
        for i in range(num_match):
            A = game[j][i][0]
            B = game[j][i][1]
            num = j + 1
            t.write("%12d %12d %12d" % (A, B, index))
            t.write("\n")
if ans:
    graph(counter,l)

t.close()
print("Configuration set created. Be ready for testing set.\n")

X =open("train_paramwX_"+str(choice)+"_"+str(noise_)+".dat","r")
b =open("train_paramwb_"+str(choice)+"_"+str(noise_)+".dat","r")
wb=np.loadtxt(b)
wX=np.loadtxt(X)
X.close()
b.close()

wX111l=[]
wX112l=[]
wX121l=[]
wX122l=[]

for i in range(num_match):
    wX111l.append(wX[l * num_match + i, 0])
    wX112l.append(wX[l * num_match + i, 1])
    wX121l.append(wX[l * num_match + i, 2])
    wX122l.append(wX[l * num_match + i, 3])
w1121=wb[l,0]
w1221=wb[l,1]
b11=wb[l,2]
b12=wb[l,3]
b21=wb[l,4]
wX111=np.array(wX111l)
wX112=np.array(wX112l)
wX121=np.array(wX121l)
wX122=np.array(wX122l)

# FASE DI TESTING
t=open("test.dat","r")
c=np.loadtxt(t)
N_test=len(strategy)*num_repetitions
s1= np.reshape(c[:,0],(num_match,N_test),order='F')
s2= np.reshape(c[:,1],(num_match,N_test),order='F')
y=[c[i,2] for i in range(0,len(c),num_match)]
a21=np.zeros(N_test)
for n in range(N_test):
    lcomb11=0
    lcomb12=0
    for i in range(num_match):
        lcomb11+=wX111[i]*s1[i,n]+wX112[i]*s2[i,n]
        lcomb12+=wX121[i]*s1[i,n]+wX122[i]*s2[i,n]
    lcomb11+=b11
    lcomb12+=b12
    a11 = 1/(1 + np.exp(-lcomb11))
    a12 = 1/(1 + np.exp(-lcomb12))
    lcomb21=w1121*a11+w1221*a12+b21
    a21[n]=1/(1 + np.exp(-lcomb21))

Accu_out=np.zeros(len(strategy))
strategy_=np.zeros(len(strategy))
for i in range(len(strategy)):
    for n in range(num_repetitions*i,(i+1)*num_repetitions,1):
        if(a21[n]>=0.5 and y[n]>=0.5):
            Accu_out[i]+=1
            strategy_[i]+=1
        if(a21[n]<=0.5 and y[n]<=0.5):
            Accu_out[i]+=1
Accu_out/=num_repetitions
Accu_out_1=(np.sum(Accu_out))/(len(strategy))
print("Testing accuracy:")
print(Accu_out)
print("Total testing accuracy:")
print(Accu_out_1)
best=np.argmax(strategy_)
print("\n")
if choice==0:
    print("If the opponent plays:")
    print(strategy[l])
    print("You should play:")
    print(strategy[best])
elif choice==1:
    print("If you play:")
    print(strategy[l])
    print("You will win against:")
    print(strategy[best])
print("You will optimize with probability:")
print(Accu_out_1*100)

t.close()