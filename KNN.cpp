#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

extern "C"
{

    struct Point
    {
        std::vector<double> features;
        std::string label;
    };

    double
    EuclideanDist(const Point &p1, const Point &p2)
    {

        double sum = 0.0;
        for (size_t i = 0; i < p1.features.size(); ++i)
        {
            sum += std::pow(p1.features[i] - p2.features[i], 2);
        }
        return std::sqrt(sum);
    }

    std::string Predict(const std::vector<Point> &data, const Point &newPoint, int k)
    {

        std::vector<std::pair<double, std::string>> distances;

        for (const Point &point : data)
        {
            double dist = EuclideanDist(point, newPoint);
            distances.push_back(std::make_pair(dist, point.label));
        }

        std::sort(distances.begin(), distances.end());

        std::map<std::string, int> top;
        for (int i = 0; i < k; i++)
        {
            top[distances[i].second]++;
        }

        int maxVote = 0;
        std::string predictedLabel = "";
        for (const auto &vote : top)
        {
            if (vote.second > maxVote)
            {
                maxVote = vote.second;
                predictedLabel = vote.first;
            }
        }
        return predictedLabel;
    }
}

int main()
{

    return 0;
}