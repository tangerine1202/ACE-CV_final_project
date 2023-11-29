#include "thread_rand.h"
#include <omp.h>
#include <algorithm>

std::vector<std::mt19937> ThreadRand::generators;
bool ThreadRand::initialised = false;

void ThreadRand::forceInit(unsigned seed)
{
    initialised = false;
    init(seed);
}

void ThreadRand::init(unsigned seed)
{
    #pragma omp critical
    {
	if(!initialised)
	{
	    unsigned nThreads = omp_get_max_threads();
	    
	    for(unsigned i = 0; i < nThreads; i++)
	    {    
		generators.push_back(std::mt19937());
		generators[i].seed(i+seed);
	    }

	    initialised = true;
	}    
    }
}

int ThreadRand::irand(int min, int max, int tid)
{
    std::uniform_int_distribution<int> dist(min, max);

    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;
    
    if(!initialised) init();
  
    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::drand(double min, double max, int tid)
{
    std::uniform_real_distribution<double> dist(min, max);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::dgauss(double mean, double stdDev, int tid)
{
    std::normal_distribution<double> dist(mean, stdDev);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

int irand(int incMin, int excMax, int tid)
{
    return ThreadRand::irand(incMin, excMax - 1, tid);
}

double drand(double incMin, double incMax,int tid)
{
    return ThreadRand::drand(incMin, incMax, tid);
}

int igauss(int mean, int stdDev, int tid)
{
    return (int) ThreadRand::dgauss(mean, stdDev, tid);
}

double dgauss(double mean, double stdDev, int tid)
{
    return ThreadRand::dgauss(mean, stdDev, tid);
}

std::vector<int> sample_indices_from_weights(const std::vector<float>& weights, int sampleSize, int tid)
{
    // ref: https://stackoverflow.com/a/53634099
    // FIXME: the reference mention that it is not efficient if the sample size is much smaller than the basic population,
    //        which is the case for our application. We may need to find a better way to do this.
    std::vector<double> vals;
    for (int iter = 0; iter < weights.size(); iter++)
    {
        double w = (double) weights[iter];
        double randVal = ThreadRand::drand(0.0, 1.0, tid);
        vals.push_back(std::pow(randVal, 1. / w));
    }

    std::vector<std::pair<int, double>> valsWithIndices;
    for (int iter = 0; iter < vals.size(); iter++)
    {
        valsWithIndices.emplace_back(iter, vals[iter]);
    }
    std::sort(valsWithIndices.begin(), valsWithIndices.end(), [](auto x, auto y) {return x.second > y.second; });

    std::vector<int> samples;
    for (auto iter = 0; iter < sampleSize; iter++)
    {
        samples.push_back(valsWithIndices[iter].first);
    }

    return samples;
}