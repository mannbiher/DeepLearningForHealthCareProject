
import train
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()  
    args = parser.parse_args()
    cv = args.cv
    train.main(cv)