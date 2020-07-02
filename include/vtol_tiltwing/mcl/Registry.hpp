#ifndef VTOL_TILTWING_REGISTRY_HPP
#define VTOL_TILTWING_REGISTRY_HPP

#include "assert.h"
#include <stdexcept>
#include <unordered_map>
#include <vtol_tiltwing/mcl/Field.hpp>
#include <iostream>

using namespace std;

class Registry {
private:
    unordered_map<string, FieldBase*> statefields;

    template <typename T>
    Field<T>* cast(FieldBase* base){
        cout << "Dynamic casting" << endl;
        Field<T>* field = dynamic_cast<Field<T>*>(base);
        if(field){
            return field;
        }
        throw runtime_error(string("Failed dynamic conversion :("));
    }

public:
    Registry();

    template <typename T>
    T get(string path){
        cout << "Getting path: " + path << endl;
        assert(statefields.find(path) != statefields.end()); // ITS FAILING HERE! IT SHOULD BE PASSING THIS ASSERT!
//        assert(("Path not found", statefields.find(path) != statefields.end()));
//        assert(("Path found", statefields.find(path) == statefields.end()));
        cout << "Passed assert" << endl;
        Field<T>* field = cast<T>(statefields[path]);
        if(field){
            T val = field->getVal();
            cout << "YELLOW" << endl;
            return val;
        }
    }

    template <typename T>
    void put(string path, T val){
        Field<T>* field = cast<T>(statefields[path]);
        if(field){
            field->setVal(val);
            return;
        }
        assert(false);
    }

    template <typename T>
    void add(string path, T initial){
        assert(statefields.find(path) == statefields.end() && "Path already exists");
//        Field<T> field = Field<T>(path, initial);
//        statefields[path] = &field;
        cout << "Adding path: " + path << endl;
        statefields[path] = new Field<T>(path, initial);
    }

};


#endif //VTOL_TILTWING_REGISTRY_HPP
