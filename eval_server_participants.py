
import requests
import numpy as np



URL_PREFIX="http://erisk.irlab.org/challenge-t1e/results/"
URL_PREFIX_RANKINGS="http://erisk.irlab.org/challenge-t1e/retrieve/"  

def penalty(delay):
    p = 0.0078
    pen = -1.0 + 2.0/(1+np.exp(-p*(delay-1)))
    return(pen)
    


def read_qrels(qrels_file):
    qrels={}
    f = open(qrels_file, 'r')
    for line in f:
        datos=line.split()
        qrels[ datos[0] ] = int(datos[1])
    f.close()
    print("\n"+str(len(qrels))+ " lines read in qrels file!\n\n")
    return(qrels)


def read_run(team_token, run_number):
    GET_REQUEST_STRING=URL_PREFIX+team_token+"/"+run_number
    print("Connecting to..."+GET_REQUEST_STRING)
    r=requests.get(GET_REQUEST_STRING)
    
    if (r.status_code==requests.codes.ok):
        print("Response..."+str(r.status_code)+" OK")
    
    
    run_results=r.json()
    print(str(len(run_results))+ " entries in the run ")
    
    return(run_results)

def n_pos(qrels):
    total_pos = 0
    
    for key in qrels:
        total_pos += qrels[key]

    return(total_pos)


def eval_performance(run_results,qrels):
    
    print("===================================================")
    print("DECISION-BASED EVALUATION:")     

    
    total_pos=n_pos(qrels)
        
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    erdes5 = np.zeros(len(run_results))
    erdes50 = np.zeros(len(run_results))
    ierdes = 0
    latency_tps = list()
    penalty_tps = list()
   
    
    for r in run_results:
        
        try:
            
            if ( qrels[ r['nick']   ] ==  r['decision'] ):
                if ( r['decision'] == 1 ): 
                    true_pos+=1
                    erdes5[ierdes]=1.0 - (1.0/(1.0+np.exp( (r['sequence']+1) - 5.0)))
                    erdes50[ierdes]=1.0 - (1.0/(1.0+np.exp( (r['sequence']+1) - 50.0)))
                    latency_tps.append(r['sequence']+1)
                    penalty_tps.append(penalty(r['sequence']+1))
                else:
                    true_neg+=1
                    erdes5[ierdes]=0
                    erdes50[ierdes]=0
            else:
                if ( r['decision'] == 1 ): 
                    false_pos+=1
                    erdes5[ierdes]=float(total_pos)/float(len(qrels))
                    erdes50[ierdes]=float(total_pos)/float(len(qrels))
                else:
                    false_neg+=1
                    erdes5[ierdes]=1
                    erdes50[ierdes]=1
                
        except KeyError:
            print("User does not appear in the qrels:"+r['nick'])
        
        ierdes+=1
    
    if ( true_pos == 0 ):
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = float(true_pos) / float(true_pos+false_pos)    
        recall = float(true_pos) / float(total_pos)   
        F1 = 2 * (precision * recall) / (precision + recall)
        
    speed = 1-np.median(np.array(penalty_tps))    
        
    print("Precision:"+str(precision))
    print("Recall:"+str(recall))
    print("F1:"+str(F1))
    print("ERDE_5:"+str(np.mean(erdes5)))
    print("ERDE_50:"+str(np.mean(erdes50)))
    print("Median latency TPs:"+str(np.median(np.array(latency_tps))))
    print("Median penalty TPs:"+str(np.median(np.array(penalty_tps))))
    print("Speed:"+str(speed))
    print("latency-weighted F1:"+str(F1*speed))
    
    
def compute_ideal_dcg_vector(qrels):
    ideal_dcg_vector=np.zeros(len(qrels))
    total_pos=n_pos(qrels)
    
    for i in range(len(qrels)):
        if i < total_pos:
            if (i==0):
                ideal_dcg_vector[i]=1.0
            else:
                ideal_dcg_vector[i]= ideal_dcg_vector[i-1] + 1.0/np.log2(i+1)
        else:
            ideal_dcg_vector[i]= ideal_dcg_vector[i-1]
        
    return(ideal_dcg_vector)
    
    
def eval_performance_rank_based(team_token, run_number, qrels):
    GET_REQUEST_STRING=URL_PREFIX_RANKINGS+team_token+"/"+run_number+"/"
    
    ideal_dcg_vector=compute_ideal_dcg_vector(qrels)
   
    print("===================================================")
    print("RANK-BASED EVALUATION:")     
    
    k=10
    
    ranks_at=[1,50,100,500,1000,2000]
    
   
    for rank in ranks_at:
        r=requests.get(GET_REQUEST_STRING+str(rank))
          
      #  if(len(r.json())==0): 
      #      break
        
        print("Analizing ranking at round "+str(rank))
        print("Rank size:"+str(len(r.json())))
        
        run_results_rank=r.json()
 
        dcg_vector=np.zeros(len(qrels))
        rels_topk=0
        
        i=0
        for r in run_results_rank:
            dcg_vector[i] = qrels[ r['nick'] ]
            
            if i>1:
                dcg_vector[i]=dcg_vector[i]/np.log2(i+1)
            if i>0:
                dcg_vector[i]=dcg_vector[i]+dcg_vector[i-1]
            if i<k:
                rels_topk+= qrels[ r['nick'] ]
        
            i+=1    
    
        ndcg_vector = dcg_vector/ideal_dcg_vector
        
        print("P@10:"+str(float(rels_topk)/10.0))
        print("NDCG@10:"+str(ndcg_vector[9]))
        print("NDCG@100:"+str(ndcg_vector[99]))

        print("===================================================")
        
        
    
   # run_results=r.json()
   # print(run_results)
   # print(str(len(run_results))+ " entries in the run ")
    
   



# parameters.... team_token   (string), 
#                run_number (integer - 0,1,2,... -)
#                gt_file (string)..path to the ground truth file
   
   
def eval_run(team_token, run_number, gt_file):
   
    qrels=read_qrels(gt_file)  
    run_results=read_run(team_token,str(run_number))
    eval_performance(run_results,qrels)
    eval_performance_rank_based(team_token,str(run_number),qrels)
    
    
    
  
