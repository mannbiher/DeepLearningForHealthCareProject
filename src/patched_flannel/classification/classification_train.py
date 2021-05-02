
import train
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("cv", help = "Cross Validation") 
    args = parser.parse_args()
    cv = args.cv
    train.main(cv)