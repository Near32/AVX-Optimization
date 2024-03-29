#ifndef RUNNINGSTATS_H
#define RUNNINGSTATS_H

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "./Mat/Mat.h"

using namespace std;

template<typename T>
class RunningStats
{
	typedef typename std::map<T,T>::iterator statIt;
	typedef typename std::map<string,T>::iterator lstatIt;
	
	typedef typename std::map<T,std::vector<T> >::iterator tStatIt;
	typedef typename std::vector<T>::iterator valIt;
	typedef typename std::map<std::string,std::vector<T> >::iterator stringStatIt;
	
	public :
	
	RunningStats(std::string file_= std::string("STATS.txt"), int valC_ = 100) : file(file_), tFile(file_+".t.txt")
	{
		counter = 0;
		valC = valC_;
	}
	
	/*
	RunningStats(std::string file_, int valC_ = 1) : file(file_), tFile(file_+".t.txt")
	{
		counter = 0;
		valC = valC_;
	}
	*/
	
	~RunningStats()
	{
	
	}
	
	inline void add(const T stat)
	{
		T val = 1;
		if( stats.count(stat) == 1)
		{
			val += stats[stat];
		}
		
		stats[stat] = val;
		
		this->writeFile();
	}
	
	
	//list add : add occurences :
	inline void ladd(const string lstat)
	{
		T val = 1;
		if( lstats.count(lstat) == 1)
		{
			val += lstats[lstat];
		}
		
		lstats[lstat] = val;
		
		this->writeFile();
	}
	
	
	inline void mean(const T stat, const T val)
	{
		T value = val;
		if( stats.count(stat) == 1)
		{
			value += stats[stat];
		}
		
		stats[stat] = value;
		
		this->writeFile();
	}
	
	//compute mean over a ladd and this function : 
	inline void lmean(const string lstat, const T val)
	{
		T value = val;
		if( lstats.count(lstat) == 1)
		{
			value += lstats[lstat];
		}
		
		lstats[lstat] = value;
		
		this->writeFile();
	}

	inline void tWriteFile()
	{
		ofstream myfile;
		myfile.open( tFile.c_str() );
		
		tStatIt it;
		
		for(it=tStats.begin();it!=tStats.end();it++)
		{
			myfile << (*it).first << " : " ;
			
			//valIt valit;
			//for(valit = (*it).second.begin() ; valit != (*it).second.end() ; valit++)
			for(int i=0; i<(*it).second.size();i++)
			{
				myfile << (*it).second[i] << " , " ;
			}
			
			myfile << std::endl;
		}
		
		stringStatIt sit;
		
		for(sit=stringtStats.begin();sit!=stringtStats.end();sit++)
		{
			myfile << (*sit).first << " : " ;
			
			//valIt valit;
			//for(valit = (*it).second.begin() ; valit != (*it).second.end() ; valit++)
			for(int i=0; i<(*sit).second.size();i++)
			{
				myfile << (*sit).second[i] << " , " ;
			}
			
			myfile << std::endl;
		}
		
		myfile.close();
	}
	
	inline void tWriteFilet()
	{
		ofstream myfile;
		myfile.open( tFile.c_str() );
		
		tStatIt it;
		stringStatIt sit;
		
		for(it=tStats.begin();it!=tStats.end();it++)
		{
			myfile << (*it).first << " : " ;
			
		}
		
		for(sit=stringtStats.begin();sit!=stringtStats.end();sit++)
		{
			myfile << (*sit).first << " : " ;
			
		}
		
		myfile << std::endl;
		
		
		bool goOn = true;
		int i = 0;
		
		while(goOn)
		{
			goOn = false;
			
			for(it=tStats.begin();it!=tStats.end();it++)
			{
				if(i<(*it).second.size())
				{
					myfile << (*it).second[i] << " : " ;
					goOn = true;
				}
				else
				{
					myfile << "ENDED" << " : " ;
				}
			}
			
			for(sit=stringtStats.begin();sit!=stringtStats.end();sit++)
			{
				if(i<(*sit).second.size())
				{
					myfile << (*sit).second[i] << " : " ;
					goOn = true;
				}
				else
				{
					myfile << "ENDED" << " : " ;
				}
			}
			
			i++;
			myfile << std::endl;
		}
		
		myfile.close();
	}
	
	void writeFile()
	{
		ofstream myfile;
		myfile.open( file.c_str() );
		
		statIt it;
		
		for(it=stats.begin();it!=stats.end();it++)
		{
			myfile << (*it).first << " = " << (*it).second << std::endl;
		}
		
		lstatIt lit;
		for(lit=lstats.begin();lit!=lstats.end();lit++)
		{
			myfile << (*lit).first << " = " << (*lit).second << std::endl;
		}
		
		myfile.close();
	}
	
	
	//temporal datas : 
	void tadd(const T& tstat, const T& val)
	{
		tStats[tstat].insert( tStats[tstat].end(), val);
		
		if(counter > valC)
		{
			//this->tWriteFile();
			this->tWriteFilet();
			counter = 0;
		}
		else
		{
			counter++;
		}
	}
	
	//temporal datas : 
	void tadd( std::string stringstat, const T& val)
	{
		stringtStats[stringstat].insert( stringtStats[stringstat].end(), val);
		
		if(counter > valC)
		{
			//this->tWriteFile();
			this->tWriteFilet();
			counter = 0;
		}
		else
		{
			counter++;
		}
	}
	
	
	
	
	private :
	
	int counter;
	int valC;
	std::string file;
	std::string tFile;
	std::map<T,T> stats;
	
	std::map<T,std::vector<T> > tStats;
	std::map<std::string,std::vector<T> > stringtStats;
	
	std::map<string,T> lstats;
};


void writeInFile(const std::string& filepath, const Mat<float>& val)
{
	ofstream myfile;
	myfile.open( filepath.c_str() );
	
	
	for(int i=1;i<=val.getLine();i++)
	{		
		for(int j=1; j<=val.getColumn();j++)
		{
			myfile << val.get(i,j) << " , " ;
		}
		
		myfile << std::endl;
	}
	
	myfile.close();
}

void writeInFile(const std::string& filepath, const std::vector<Mat<float> >& val)
{
	ofstream myfile;
	myfile.open( filepath.c_str() );
	
	for(int k=0;k<val.size();k++)
	{
		for(int i=1;i<=val[k].getLine();i++)
		{		
			for(int j=1; j<=val[k].getColumn();j++)
			{
				myfile << val[k].get(i,j) << " , " ;
			}
		
			myfile << std::endl;
		}
		
		//myfile << std::endl << std::endl;
	}	
	myfile.close();
}

#endif
