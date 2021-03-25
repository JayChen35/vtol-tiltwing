#ifndef VTOL_TILTWING_FLAG_HPP
#define VTOL_TILTWING_FLAG_HPP

#include "assert.h"
#include <string>
#include <unordered_map>
#include <Logger.hpp>

using namespace std;

class Flag {
private:
    unordered_map<string, string> flags;

    string get(string path){
        assert(flags.find(path) != flags.end());
        Logger::log("Passed assert", LogSeverity::DEBUG);
        string var = flags[path];
        return var;
    }

public:
    Flag();

    void get(string path, string &out){
        out = get(path);
    }

    void get(string path, int &out){
        out = stoi(get(path));
    }

    void get(string path, float &out){
        out = stof(get(path));
    }

    void get(string path, long &out){
        out = stol(get(path));
    }

    void get(string path, bool &out){
        string temp = get(path);
        if(temp == "0"){ out = false; }
        else{ out = true; }
    }

    void put(string path, string val){
        flags[path] = val;
    }

    void put(string path, int val){
        flags[path] = to_string(val);
    }

    void put(string path, float val){
        flags[path] = to_string(val);
    }

    void put(string path, long val){
        flags[path] = to_string(val);
    }

    void put(string path, bool val){
        flags[path] = to_string(val);
    }

};

#endif //VTOL_TILTWING_FLAG_HPP
