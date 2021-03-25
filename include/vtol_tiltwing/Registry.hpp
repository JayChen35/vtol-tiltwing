#ifndef VTOL_TILTWING_REGISTRY_HPP
#define VTOL_TILTWING_REGISTRY_HPP

#include "assert.h"
#include <string>
#include <unordered_map>
#include <Logger.hpp>

using namespace std;

class Registry {
private:
    unordered_map<string, string> statefields;

    string get(string path){
        assert(statefields.find(path) != statefields.end());
        Logger::log("Passed assert", LogSeverity::DEBUG);
        string var = statefields[path];
        return var;
    }

public:
    Registry();

    void get(string path, string& out){
        out = get(path);
    }

    void get(string path, int& out){
        out = stoi(get(path));
    }

    void get(string path, float& out){
        out = stof(get(path));
    }

    void get(string path, long& out){
        out = stol(get(path));
    }

    void get(string path, bool& out){
        string temp = get(path);
        if(temp == "0"){ out = false; }
        else{ out = true; }
    }

    void put(string path, string val){
        statefields[path] = val;
    }

    void put(string path, int val){
        statefields[path] = to_string(val);
    }

    void put(string path, float val){
        statefields[path] = to_string(val);
    }

    void put(string path, long val){
        statefields[path] = to_string(val);
    }

    void put(string path, bool val){
        statefields[path] = to_string(val);
    }

};


#endif //VTOL_TILTWING_REGISTRY_HPP
