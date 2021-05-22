import networkx as nx
import pandas as pd
import csv
from operator import itemgetter


def graph_from_tsv(path):
    """
    From a tsv file generate a graph (unweighted graph)

    :param path:
    :return:
    """
    input_file_handler = open(path, 'r', encoding="utf-8")
    tsv_reader = csv.reader(input_file_handler, delimiter='\t')
    tsv_reader.__next__()
    adj_list = []
    for pokemon in tsv_reader:
        pokemon1 = pokemon[0]
        pokemon2 = pokemon[1]
        adj_list.append((pokemon1, pokemon2))
    input_file_handler.close()

    graph = nx.Graph()
    graph.add_edges_from(adj_list)

    return graph


def compute_top_k(map__node_id__score, remove_pokemon, k=20):
    """
    Get top k nodes
    :param remove_pokemon: Pokemons that we want to remove from the top k
    """
    list__node_id__score = [(node_id, score) for node_id, score in map__node_id__score.items() if node_id not in remove_pokemon]
    list__node_id__score.sort(key=lambda x: (-x[1], x[0]))
    return list__node_id__score[:k]


def generate_topic(pokemon_set, graph, neighbours=False):
    """
    Function that will return a topic dictionary based on the pokemon set given in inputs. Pokemons related with this
    topic will have a value of 1

    :param pokemon_set: Topics (list or set)
    :param graph: Graph from which neighbours will be fetched
    :param neighbours: Boolean. If set to False we will consider topic only the provided pokemon sets. If set to True
        we will consider the topic all the pokemons that have high affinity to those of the pokemon set
    :return: Dict
    """
    topic_dict = dict.fromkeys(list(graph.nodes), 0)
    # Loop over the pokemon sets
    if neighbours:
        for pokemon in pokemon_set:
            # Get all the pokemons that have high affinity with the pokemons in the input
            topic = list(graph.neighbors(pokemon))
            # Add a 1 to those pokemons in the general dictionary
            for i in topic:
                topic_dict[i] = 1
    else:
        for pokemon in pokemon_set:
            topic_dict[pokemon] = 1

    return topic_dict


def compute_good_local_community(graph, seed_node_id, alpha=0.9):
    #
    # Creation of the teleporting probability distribution for the selected node...
    map_teleporting_probability_distribution__node_id__probability = {}
    for node_id in graph:
        map_teleporting_probability_distribution__node_id__probability[node_id] = 0.
    map_teleporting_probability_distribution__node_id__probability[seed_node_id] = 1.
    #
    # Computation of the PageRank vector.
    map__node_id__node_pagerank_value = nx.pagerank(graph, alpha=alpha,
                                                    personalization=map_teleporting_probability_distribution__node_id__probability)
    #
    # Put all nodes in a list and sort the list in descending order of the normalized_score
    sorted_list__node_id__normalized_score = [(node_id, score / graph.degree[node_id])
                                              for node_id, score in map__node_id__node_pagerank_value.items()]
    sorted_list__node_id__normalized_score.sort(key=lambda x: (-x[1], x[0]))
    #
    # LET'S SWEEP!
    index_representing_the_set_of_node_ids_with_maximum_conductance = -1
    min_conductance_value = float("+inf")
    set__node_ids_in_the_candidate_community = set()
    set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes = set(graph.nodes())
    for sweep_index in range(0, len(sorted_list__node_id__normalized_score) - 1):
        #
        # Creation of the set of nodes representing the candidate community and
        # its complement to the entire set of nodes in the graph.
        current_node_id = sorted_list__node_id__normalized_score[sweep_index][0]
        set__node_ids_in_the_candidate_community.add(current_node_id)
        set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes.remove(current_node_id)
        #
        # Evaluation of the quality of the candidate community according to its conductance value.
        conductance_value = nx.algorithms.cuts.conductance(graph,
                                                           set__node_ids_in_the_candidate_community,
                                                           set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes)
        #
        # Discard local communities with conductance 0 or 1.
        if conductance_value == 0. or conductance_value == 1.:
            continue

        # Discard communities with more than 140 pokemons
        if len(set__node_ids_in_the_candidate_community) > 140:
            continue
        #
        # Update the values of variables representing the best solution generated so far.
        if conductance_value < min_conductance_value:
            min_conductance_value = conductance_value
            index_representing_the_set_of_node_ids_with_maximum_conductance = sweep_index
    #
    # Creation of the set of nodes representing the best local community generated by the sweeping procedure.
    set__node_ids_with_minimum_conductance = set([node_id for node_id, normalized_score in
                                                  sorted_list__node_id__normalized_score[
                                                  :index_representing_the_set_of_node_ids_with_maximum_conductance + 1]])
    #
    return set__node_ids_with_minimum_conductance, min_conductance_value


def main_part_1():
    """
    Run code for exercise 2.1
    """
    # Construct graph
    graph = graph_from_tsv('data/pkmn_graph_data.tsv')

    # Select the size of the teams
    team_size = 6
    # Specify mode in which we want to create the topics (topics will be based only on the initial team members)
    topic_generator_with_neighbours=False

    set_A = ['Pikachu']
    set_B = ['Venusaur', 'Charizard', 'Blastoise']
    set_C = ['Excadrill', 'Dracovish', 'Whimsicott', 'Milotic']

    sets = {'set_A': set_A,
            'set_B': set_B,
            'set_C': set_C}

    teams = {}
    # Loop over all the set of teams
    for key, values in sets.items():
        # Create dictionary setting 1 for the pokemons that belong to the topic)
        topic = generate_topic(values, graph, neighbours=topic_generator_with_neighbours)
        # Compute page ranke
        pagerank = nx.pagerank(graph, personalization=topic, alpha=.33)
        # Based on the page rank get the top 6 pokemons (excluding those pokemons that are already part of the initial
        # team)
        topk = compute_top_k(pagerank, remove_pokemon=values, k=team_size-len(values))
        teams['team_' + key[-1]] = set([i[0] for i in topk] + values)

    # Print results
    print('Set of Pokemons using Set_A')
    print(teams['team_A'])
    print('==================')
    print('Set of Pokemons using Set_B')
    print(teams['team_B'])
    print('==================')
    print('Set of Pokemons using Set_C')
    print(teams['team_C'])
    print('==================')
    print('==================')

    # We will now perform the same analysis but with a new set of pokemons
    set_1 = ['Charizard']
    set_2 = ['Venusaur']
    set_3 = ['Kingdra']
    set_4 = ['Charizard', 'Venusaur']
    set_5 = ['Charizard', 'Kingdra']
    set_6 = ['Venusaur', 'Kingdra']

    sets = {'set_1': set_1,
            'set_2': set_2,
            'set_3': set_3,
            'set_4': set_4,
            'set_5': set_5,
            'set_6': set_6}

    teams = {}
    # Loop over all the sets of teams
    for key, values in sets.items():
        topic = generate_topic(values, graph)
        pagerank = nx.pagerank(graph, personalization=topic, alpha=.33)
        topk = compute_top_k(pagerank, remove_pokemon=values, k=team_size - len(values))
        teams['team_' + key[-1]] = set([i[0] for i in topk] + values)

    print('Set of Pokemons using Charizard')
    print(teams['team_1'])
    print('==================')
    print('Set of Pokemons using Venusaur')
    print(teams['team_2'])
    print('==================')
    print('Set of Pokemons using Kingdra')
    print(teams['team_3'])
    print('==================')
    print('Set of Pokemons using Charizard and Venusaur')
    print(teams['team_4'])
    print('==================')
    print('Set of Pokemons using Charizard and Kingdra')
    print(teams['team_5'])
    print('==================')
    print('Set of Pokemons using Venusaur and Kingdra')
    print(teams['team_6'])
    print('==================')
    print('==================')

    # Finally we will look at the intersections between the teams
    print('Number of team members inside the Team(Charizard, Venusaur) that '
          'are neither in Team(Charizard) nor in Team(Venusaur)')
    print(len(teams['team_4'].difference(teams['team_1'].union(teams['team_2']))))
    print('==================')
    print('Number of team members inside the Team(Charizard, Kingdra) that '
          'are neither in Team(Charizard) nor in Team(Kingdra)')
    print(len(teams['team_5'].difference(teams['team_1'].union(teams['team_3']))))
    print('==================')
    print('Number of team members inside the Team(Venusaur, Kingdra) that '
          'are neither in Team(Venusaur) nor in Team(Kingdra)')
    print(len(teams['team_6'].difference(teams['team_2'].union(teams['team_3']))))


def main_part_2():
    """
    Run code for exercise 2.2
    """
    graph = graph_from_tsv('data/pkmn_graph_data.tsv')
    output_path = 'data/output.tsv'

    # List of alpha values we will investigate
    damping_factors = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3,
                       0.25, 0.2, 0.15, 0.1, 0.05]

    # Set initial lists and dictionaries which we will be filling out throughout the next loop
    pokemons = list(graph.nodes)
    pokemon_frequency = dict.fromkeys(list(graph.nodes), 0)

    # We will reuse the code provided to use during the Lab. We will adjust the code such that we force the communities
    # to have a cardinality of at least 140 pokemons.

    # List in which we will store all the final results (already with the best community)
    results = []
    for j, pokemon in enumerate(pokemons):
        print(j/len(pokemons))
        # Temporary list of tuples in which we will have the nodes of each community and the conductance value (we will
        # end up choosing the minimum conductance)
        temp = []
        for factor in damping_factors:
            temp.append((compute_good_local_community(graph=graph, seed_node_id=pokemon, alpha=factor)))

        best_factor = min(temp, key=lambda t: t[1])
        results.append([pokemon, len(best_factor[0]), best_factor[0], best_factor[1]])
        # We will count how many pokemons there are in each community
        for i in best_factor[0]:
            pokemon_frequency[i] = pokemon_frequency[i] + 1

    results_sorted = sorted(results, key=itemgetter(0))

    # Write all the results into a final tsv file
    print('...writing tsv file...')
    with open(output_path, 'w', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['pokemon_name', 'number_of_nodes_in_the_local_comunity',
                             'conductance_value_of_the_local_comunity'])
        for row in results_sorted:
            tsv_writer.writerow([row[0], row[1], row[3]])

    sort_dict = dict(sorted(pokemon_frequency.items(), key=lambda x: x[1], reverse=True))
    df = pd.DataFrame.from_dict(sort_dict, orient='index', columns=['Frequency'])
    top5 = df.head(5)
    tail5 = df.tail(5)

    top5.to_excel('top5.xlsx')
    tail5.to_excel('tail5.xlsx')
    print(top5)
    print(tail5)


if __name__:
    main_part_1()
    main_part_2()

